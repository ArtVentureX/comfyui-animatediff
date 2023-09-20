import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange

import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
from comfy.model_base import BaseModel
from comfy.ldm.modules.attention import SpatialTransformer
from nodes import KSampler

from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .sliding_schedule import ContextSchedules
from .sliding_context_sampling import SlidingContext, inject_sampling_function, eject_sampling_function


SLIDING_CONTEXT_LENGTH = 16


def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None):
    for layer in ts:
        if isinstance(layer, openaimodel.TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, VanillaTemporalModule):
            x = layer(x, context)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            transformer_options["current_index"] += 1
        elif isinstance(layer, openaimodel.Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


def groupnorm_mm_factory(video_length: int):
    def groupnorm_mm_forward(self, input: Tensor) -> Tensor:
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        axes_factor = input.size(0) // video_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input

    return groupnorm_mm_forward


orig_forward_timestep_embed = openaimodel.forward_timestep_embed
orig_maximum_batch_area = model_management.maximum_batch_area
orig_groupnorm_forward = torch.nn.GroupNorm.forward
openaimodel.forward_timestep_embed = forward_timestep_embed


def inject_motion_module_to_unet_legacy(unet, motion_module: MotionWrapper):
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1])

    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].insert(-1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
        else:
            unet.output_blocks[unet_idx].append(motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
    if motion_module.is_v2:
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0])

    unet.motion_module = motion_module


def eject_motion_module_from_unet_legacy(unet):
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    for unet_idx in range(12):
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)

    if unet.motion_module.is_v2:
        unet.middle_block.pop(-2)

    del unet.motion_module


def inject_motion_module_to_unet(unet, motion_module: MotionWrapper):
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1])

    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].insert(-1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
        else:
            unet.output_blocks[unet_idx].append(motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
    if motion_module.is_v2:
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0])

    unet.motion_module = motion_module


def eject_motion_module_from_unet(unet):
    for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
        unet.input_blocks[unet_idx].pop(-1)

    for unet_idx in range(12):
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)

    if unet.motion_module.is_v2:
        unet.middle_block.pop(-2)

    del unet.motion_module


injectors = {
    "legacy": inject_motion_module_to_unet_legacy,
    "default": inject_motion_module_to_unet,
}

ejectors = {
    "legacy": eject_motion_module_from_unet_legacy,
    "default": eject_motion_module_from_unet,
}


class AnimateDiffSlidingWindowOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": SLIDING_CONTEXT_LENGTH, "min": 2, "max": 32}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 32}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 32}),
                "context_schedule": (ContextSchedules.CONTEXT_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SLIDING_WINDOW_OPTS",)
    FUNCTION = "init_options"
    CATEGORY = "Animate Diff"

    def init_options(self, context_length, context_stride, context_overlap, context_schedule, closed_loop):
        ctx = SlidingContext(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=context_schedule,
            closed_loop=closed_loop,
        )

        return (ctx,)


class AnimateDiffSampler(KSampler):
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "motion_module": ("MOTION_MODULE",),
                "inject_method": (["default", "legacy"],),
                "frame_number": (
                    "INT",
                    {"default": 16, "min": 2, "max": 10000, "step": 1},
                ),
            }
        }
        inputs["required"].update(KSampler.INPUT_TYPES()["required"])
        inputs["optional"] = {"sliding_window_opts": ("SLIDING_WINDOW_OPTS",)}
        return inputs

    FUNCTION = "animatediff_sample"
    CATEGORY = "Animate Diff"

    def __init__(self) -> None:
        super().__init__()
        self.prev_beta = None
        self.prev_linear_start = None
        self.prev_linear_end = None

    def override_beta_schedule(self, model: BaseModel):
        logger.info(f"Override beta schedule.")
        self.prev_beta = model.get_buffer("betas").cpu().clone()
        self.prev_linear_start = model.linear_start
        self.prev_linear_end = model.linear_end
        model.register_schedule(
            given_betas=None,
            beta_schedule="sqrt_linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.012,
            cosine_s=8e-3,
        )

    def restore_beta_schedule(self, model: BaseModel):
        logger.info(f"Restoring beta schedule.")
        model.register_schedule(
            given_betas=self.prev_beta,
            linear_start=self.prev_linear_start,
            linear_end=self.prev_linear_end,
        )
        self.prev_beta = None
        self.prev_linear_start = None
        self.prev_linear_end = None

    def inject_motion_module(self, model, motion_module: MotionWrapper, inject_method: str, frame_number: int):
        model = model.clone()
        unet = model.model.diffusion_model

        logger.info(f"Injecting motion module with method {inject_method}.")
        motion_module.set_video_length(frame_number)
        injectors[inject_method](unet, motion_module)
        self.override_beta_schedule(model.model)
        if not motion_module.is_v2:
            logger.info(f"Hacking GroupNorm.forward function.")
            torch.nn.GroupNorm.forward = groupnorm_mm_factory(frame_number)

        return model

    def inject_sliding_sampler(self, video_length, sliding_window_opts: SlidingContext = None):
        ctx = sliding_window_opts.copy() if sliding_window_opts else SlidingContext()
        ctx.video_length = video_length

        inject_sampling_function(ctx)

    def eject_motion_module(self, model, inject_method):
        unet = model.model.diffusion_model

        self.restore_beta_schedule(model.model)
        if not unet.motion_module.is_v2:
            logger.info(f"Restore GroupNorm.forward function.")
            torch.nn.GroupNorm.forward = orig_groupnorm_forward

        logger.info(f"Ejecting motion module with method {inject_method}.")
        ejectors[inject_method](unet)

    def eject_sliding_sampler(self):
        eject_sampling_function()

    def animatediff_sample(
        self,
        motion_module,
        inject_method,
        frame_number,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        sliding_window_opts: SlidingContext = None,
        **kwargs,
    ):
        # init latents
        samples = latent_image["samples"]
        init_frames = len(samples)
        if init_frames < frame_number:
            # TODO: apply different noise to each frame
            last_frame = samples[-1].clone().cpu().unsqueeze(0)
            repeated_last_frames = last_frame.repeat(frame_number - init_frames, 1, 1, 1)
            samples = torch.cat((samples, repeated_last_frames), dim=0)

        latent_image = {"samples": samples}

        # validate context_length
        context_length = sliding_window_opts.context_length if sliding_window_opts else SLIDING_CONTEXT_LENGTH
        is_sliding = frame_number > context_length
        video_length = context_length if is_sliding else frame_number

        if video_length > motion_module.encoding_max_len:
            error = f'{"context_length" if is_sliding else "frame_number"} = {video_length}'
            raise ValueError(
                f"AnimateDiff model {motion_module.mm_type} has upper limit of {motion_module.encoding_max_len} frames, but received {error}."
            )

        # inject sliding sampler
        if is_sliding:
            self.inject_sliding_sampler(frame_number, sliding_window_opts=sliding_window_opts)

        # inject motion module
        model = self.inject_motion_module(model, motion_module, inject_method, video_length)

        try:
            return super().sample(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                **kwargs,
            )
        except:
            raise
        finally:
            self.eject_motion_module(model, inject_method)
            if is_sliding:
                self.eject_sliding_sampler()

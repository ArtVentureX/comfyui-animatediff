import os
import json
import torch
import numpy as np
import hashlib
from typing import Dict, List
from torch import Tensor
from torch.nn.functional import group_norm
from PIL import Image, ImageSequence
from PIL.PngImagePlugin import PngInfo
from einops import rearrange

import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
from comfy.model_base import BaseModel
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.utils import load_torch_file, calculate_parameters
from nodes import KSampler

from .logger import logger
from .motion_module import MotionWrapper, VanillaTemporalModule
from .model_utils import get_available_models, get_model_path, get_model_hash
from .utils import pil2tensor


def forward_timestep_embed(
    ts, x, emb, context=None, transformer_options={}, output_shape=None
):
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
        input = group_norm(input, self.num_groups,
                           self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input

    return groupnorm_mm_forward


orig_forward_timestep_embed = openaimodel.forward_timestep_embed
orig_maximum_batch_area = model_management.maximum_batch_area
orig_groupnorm_forward = torch.nn.GroupNorm.forward
openaimodel.forward_timestep_embed = forward_timestep_embed

motion_modules: Dict[str, MotionWrapper] = {}


def load_motion_module(model_name: str):
    model_path = get_model_path(model_name)
    model_hash = get_model_hash(model_path)
    if model_hash not in motion_modules:
        logger.info(f"Loading motion module {model_name}")
        mm_state_dict = load_torch_file(model_path)
        motion_module = MotionWrapper.from_pretrained(
            mm_state_dict, model_name)

        params = calculate_parameters(mm_state_dict, "")
        if model_management.should_use_fp16(model_params=params):
            logger.info(f"Converting motion module to fp16.")
            motion_module.half()
        offload_device = model_management.unet_offload_device()
        motion_module = motion_module.to(offload_device)

        motion_modules[model_hash] = motion_module

    return motion_modules[model_hash]


def inject_motion_module_to_unet_legacy(unet, motion_module: MotionWrapper):
    for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 2 == 2:
            unet.output_blocks[unet_idx].insert(
                -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
        else:
            unet.output_blocks[unet_idx].append(
                motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
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
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    for unet_idx in range(12):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if unet_idx % 3 == 2 and unet_idx != 11:
            unet.output_blocks[unet_idx].insert(
                -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
        else:
            unet.output_blocks[unet_idx].append(
                motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
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


class AnimateDiffModuleLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_models(),),
            },
        }

    RETURN_TYPES = ("MOTION_MODULE",)
    CATEGORY = "Animate Diff"
    FUNCTION = "load_motion_module"

    def load_motion_module(
        self,
        model_name: str,
    ):
        motion_module = load_motion_module(model_name)

        return (motion_module,)


class AnimateDiffSampler(KSampler):
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "motion_module": ("MOTION_MODULE",),
                "inject_method": (["default", "legacy"],),
                "frame_number": (
                    "INT",
                    {"default": 16, "min": 2, "max": 32, "step": 1},
                ),
            }
        }
        inputs["required"].update(KSampler.INPUT_TYPES()["required"])
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

    def inject_motion_module(
        self, model, motion_module: MotionWrapper, inject_method: str, frame_number: int
    ):
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

    def eject_motion_module(self, model, inject_method):
        unet = model.model.diffusion_model

        self.restore_beta_schedule(model.model)
        if not unet.motion_module.is_v2:
            logger.info(f"Restore GroupNorm.forward function.")
            torch.nn.GroupNorm.forward = orig_groupnorm_forward

        logger.info(f"Ejecting motion module with method {inject_method}.")
        ejectors[inject_method](unet)

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
    ):
        model = self.inject_motion_module(
            model, motion_module, inject_method, frame_number
        )

        init_frames = len(latent_image["samples"])
        samples = latent_image["samples"][:init_frames, :, :, :].clone().cpu()

        if init_frames < frame_number:
            last_frame = samples[-1].unsqueeze(0)
            repeated_last_frames = last_frame.repeat(
                frame_number - init_frames, 1, 1, 1
            )
            samples = torch.cat((samples, repeated_last_frames), dim=0)

        latent_image = {"samples": samples}

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
            )
        except:
            raise
        finally:
            self.eject_motion_module(model, inject_method)


class AnimateDiffCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "save_image": ([True, False],),
                "filename_prefix": ("STRING", {"default": "animate_diff"}),
                "format": (["image/gif", "image/webp"] +
                           ["video/"+x[:-5] for x in folder_paths.get_filename_list("video_formats")],),
                "pingpong": ([False, True],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Animate Diff"
    FUNCTION = "generate_gif"

    def generate_gif(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        save_image=True,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        # convert images to numpy
        frames: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)

        # save image
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, file)
        frames[0].save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        if pingpong:
            frames = frames + frames[-2:0:-1]

        format_type, format_ext = format.split("/")

        if format_type == "image":
            file = f"{filename}_{counter:05}_.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
        else:
            # save webm
            import shutil
            import subprocess

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                raise ProcessLookupError("Could not find ffmpeg")
            video_format_path = folder_paths.get_full_path(
                "video_formats", format_ext + ".json")
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{frames[0].width}x{frames[0].height}"
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                + video_format['main_pass'] + [file_path]

            env = os.environ
            if "environment" in video_format:
                env.update(video_format["environment"])
            with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image else "temp",
                "format": format,
            }
        ]
        return {"ui": {"videos": previews}}


class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.join(folder_paths.get_input_directory(), "video")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)

        files = [f"video/{f}" for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]

        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
            },
            "optional": {
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "step": 1}),
                "frame_limit": ("INT", {"default": 16, "min": 1, "max": 10240, "step": 1}),
            }
        }

    CATEGORY = "Animate Diff/Utils"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "load"

    def load_gif(self, gif_path: str, frame_start: int, frame_limit: int):
        image = Image.open(gif_path)
        frames = []

        for i, frame in enumerate(ImageSequence.Iterator(image)):
            if i < frame_start:
                continue
            elif i >= frame_start + frame_limit:
                break
            else:
                frames.append(pil2tensor(frame.copy().convert("RGB")))

        return frames

    def load_video(self, video_path, frame_start: int, frame_limit: int):
        import cv2

        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        frames = []
        for i in range(frame_limit):
            # Read the next frame
            ret, frame = video.read()
            if ret:
                # Convert the frame to RGB (OpenCV uses BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the NumPy array to a PIL image and append to list
                frames.append(pil2tensor(Image.fromarray(frame)))
            else:
                break

        video.release()

        return frames

    def load(self, video: str, frame_start=0, frame_limit=16):
        print("path", video)
        video_path = folder_paths.get_annotated_filepath(video)
        (_, ext) = os.path.splitext(video_path)

        if ext.lower() in {".gif", ".webp"}:
            frames = self.load_gif(video_path, frame_start, frame_limit)
        elif ext.lower() in {".webp", ".mp4", ".mov", ".avi"}:
            frames = self.load_video(video_path, frame_start, frame_limit)
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        return (torch.cat(frames, dim=0),)

    @classmethod
    def IS_CHANGED(s, image, *args, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, video, *args, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)

        return True


class ImageSizeAndBatchSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    CATEGORY = "Animate Diff/Utils"
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size")
    FUNCTION = "batch_size"

    def batch_size(self, image: Tensor):
        (batch_size, width, height) = image.shape[0:3]
        return (width, height, batch_size)


class ImageChunking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "chunk_size": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1}),
                "allow_remainder": ([True, False],),
            },
        }

    CATEGORY = "Animate Diff/Utils"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "chunk"

    def chunk(self, images: Tensor, chunk_size: int, allow_remainder: bool):
        # Check if tensor is divisible into chunks of chunk_size
        if images.shape[0] % chunk_size != 0 and not allow_remainder:
            raise ValueError(
                "Tensor's first dimension is not divisible by chunk size")

        # Use torch.chunk to divide the tensor
        chunk_count = images.shape[0] // chunk_size + \
            images.shape[0] % chunk_size

        print("chunk_count", chunk_count)
        chunks = torch.chunk(images, chunk_count, dim=0)

        return (list(chunks), )


NODE_CLASS_MAPPINGS = {
    "AnimateDiffModuleLoader": AnimateDiffModuleLoader,
    "AnimateDiffCombine": AnimateDiffCombine,
    "AnimateDiffSampler": AnimateDiffSampler,
    "LoadVideo": LoadVideo,
    "ImageSizeAndBatchSize": ImageSizeAndBatchSize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateDiffModuleLoader": "Animate Diff Module Loader",
    "AnimateDiffSampler": "Animate Diff Sampler",
    "AnimateDiffCombine": "Animate Diff Combine",
    "LoadVideo": "Load Video",
    "ImageBatchSize": "Get Image Size + Batch Size",
}

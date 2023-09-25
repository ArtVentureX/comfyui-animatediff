import os
import json
import torch
import numpy as np
import hashlib
from typing import List, Dict
from torch import Tensor
from PIL import Image, ImageSequence
from PIL.PngImagePlugin import PngInfo

import folder_paths

from .model_utils import get_available_models, load_motion_module, get_available_loras, load_lora
from .utils import pil2tensor, ensure_opencv
from .sampler import AnimateDiffSampler, AnimateDiffSlidingWindowOptions


SLIDING_CONTEXT_LENGTH = 16

video_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")
video_formats = ["video/" + x[:-5] for x in os.listdir(video_formats_dir)]


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


class AnimateDiffLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (get_available_loras(),),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lora_stack": ("MOTION_LORA_STACK",),
            },
        }

    RETURN_TYPES = ("MOTION_LORA_STACK",)
    CATEGORY = "Animate Diff"
    FUNCTION = "load_lora"

    def load_lora(
        self,
        lora_name: str,
        alpha: float,
        lora_stack: List = None,
    ):
        if not lora_stack:
            lora_stack = []

        lora = load_lora(lora_name)
        lora_stack.append((lora, alpha))

        return (lora_stack,)


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
                "save_image": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "animate_diff"}),
                "format": (["image/gif", "image/webp"] + video_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
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
        output_dir = folder_paths.get_output_directory() if save_image else folder_paths.get_temp_directory()
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
            video_format_path = os.path.join(video_formats_dir, format_ext + ".json")
            with open(video_format_path, "r") as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{frames[0].width}x{frames[0].height}"
            args = (
                [
                    ffmpeg_path,
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    dimensions,
                    "-r",
                    str(frame_rate),
                    "-i",
                    "-",
                ]
                + video_format["main_pass"]
                + [file_path]
            )

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

        files = [f"video/{f}" for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
            },
            "optional": {
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "frame_limit": ("INT", {"default": 16, "min": 1, "max": 10240, "step": 1}),
            },
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
        ensure_opencv()
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
        video_path = folder_paths.get_annotated_filepath(video)
        (_, ext) = os.path.splitext(video_path)

        if ext.lower() in {".gif", ".webp"}:
            frames = self.load_gif(video_path, frame_start, frame_limit)
        elif ext.lower() in {".webp", ".mp4", ".mov", ".avi", ".webm"}:
            frames = self.load_video(video_path, frame_start, frame_limit)
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        return (torch.cat(frames, dim=0), len(frames))

    @classmethod
    def IS_CHANGED(s, image, *args, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
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
        (batch_size, height, width) = image.shape[0:3]
        return (width, height, batch_size)


class ImageChunking:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "chunk_size": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1}),
                "allow_remainder": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "Animate Diff/Utils"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "chunk"

    def chunk(self, images: Tensor, chunk_size: int, allow_remainder: bool):
        # Check if tensor is divisible into chunks of chunk_size
        if images.shape[0] % chunk_size != 0 and not allow_remainder:
            raise ValueError("Tensor's first dimension is not divisible by chunk size")

        # Use torch.chunk to divide the tensor
        chunk_count = images.shape[0] // chunk_size + images.shape[0] % chunk_size

        print("chunk_count", chunk_count)
        chunks = torch.chunk(images, chunk_count, dim=0)

        return (list(chunks),)


NODE_CLASS_MAPPINGS = {
    "AnimateDiffModuleLoader": AnimateDiffModuleLoader,
    "AnimateDiffLoraLoader": AnimateDiffLoraLoader,
    "AnimateDiffCombine": AnimateDiffCombine,
    "AnimateDiffSampler": AnimateDiffSampler,
    "AnimateDiffSlidingWindowOptions": AnimateDiffSlidingWindowOptions,
    "LoadVideo": LoadVideo,
    "ImageSizeAndBatchSize": ImageSizeAndBatchSize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateDiffModuleLoader": "Animate Diff Module Loader",
    "AnimateDiffLoraLoader": "Animate Diff Lora Loader",
    "AnimateDiffSampler": "Animate Diff Sampler",
    "AnimateDiffSlidingWindowOptions": "Sliding Window Options",
    "AnimateDiffCombine": "Animate Diff Combine",
    "LoadVideo": "Load Video",
    "ImageSizeAndBatchSize": "Get Image Size + Batch Size",
}

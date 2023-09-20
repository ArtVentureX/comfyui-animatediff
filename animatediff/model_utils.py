import os
import hashlib
from typing import Dict

import folder_paths
import comfy.model_management as model_management
from comfy.utils import load_torch_file, calculate_parameters

from .logger import logger
from .motion_module import MotionWrapper


motion_modules: Dict[str, MotionWrapper] = {}


folder_paths.folder_names_and_paths["AnimateDiff"] = (
    [
        os.path.join(folder_paths.models_dir, "AnimateDiff"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    ],
    folder_paths.supported_pt_extensions,
)
folder_paths.folder_names_and_paths["video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"],
)


def get_available_models():
    return folder_paths.get_filename_list("AnimateDiff")


def get_model_path(model_name):
    return folder_paths.get_full_path("AnimateDiff", model_name)


def get_model_hash(file_path):
    with open(file_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        return hashlib.sha256(bytes).hexdigest()


def load_motion_module(model_name: str):
    model_path = get_model_path(model_name)
    model_hash = get_model_hash(model_path)
    if model_hash not in motion_modules:
        logger.info(f"Loading motion module {model_name}")
        mm_state_dict = load_torch_file(model_path)
        motion_module = MotionWrapper.from_state_dict(mm_state_dict, model_name)

        params = calculate_parameters(mm_state_dict, "")
        if model_management.should_use_fp16(model_params=params):
            logger.info(f"Converting motion module to fp16.")
            motion_module.half()
        offload_device = model_management.unet_offload_device()
        motion_module = motion_module.to(offload_device)

        motion_modules[model_hash] = motion_module

    return motion_modules[model_hash]

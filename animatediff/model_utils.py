import os
import hashlib

import folder_paths

from .logger import logger


folder_paths.folder_names_and_paths["AnimateDiff"] = (
    [
        os.path.join(folder_paths.models_dir, "AnimateDiff"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    ],
    folder_paths.supported_pt_extensions,
)

known_models = {
    "aa7fd8a200a89031edd84487e2a757c5315460eca528fa70d4b3885c399bffd5": "mm_sd_v14.ckpt",
    "cf16ea656cb16124990c8e2c70a29c793f9841f3a2223073fac8bd89ebd9b69a": "mm_sd_v15.ckpt",
    "0aaf157b9c51a0ae07cb5d9ea7c51299f07bddc6f52025e1f9bb81cd763631df": "mm-Stabilized_high.pth",
    "39de8b71b1c09f10f4602f5d585d82771a60d3cf282ba90215993e06afdfe875": "mm-Stabilized_mid.pth",
    "3cb569f7ce3dc6a10aa8438e666265cb9be3120d8f205de6a456acf46b6c99f4": "temporaldiff-v1-animatediff.ckpt",
    "69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0": "mm_sd_v15_v2.ckpt",
}

v2_models = ["69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0"]


def get_available_models():
    return folder_paths.get_filename_list("AnimateDiff")


def get_model_path(model_name):
    return folder_paths.get_full_path("AnimateDiff", model_name)


def sha256_file(file_path):
    with open(file_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        return hashlib.sha256(bytes).hexdigest()


def validate_mm_model(model_name):
    model_path = get_model_path(model_name)
    model_hash = sha256_file(model_path)

    if model_hash in known_models:
        logger.info(f"You are using {model_name}, which has been tested and supported.")
    else:
        logger.warn(
            f"Your model {model_name} has not been tested and supported."
            "Either your download is incomplete or your model has not been tested. "
            "Please use at your own risk."
        )

    using_v2 = model_hash in v2_models

    return (model_hash, using_v2)
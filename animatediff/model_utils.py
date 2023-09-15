import os
import hashlib

import folder_paths


folder_paths.folder_names_and_paths["AnimateDiff"] = (
    [
        os.path.join(folder_paths.models_dir, "AnimateDiff"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    ],
    folder_paths.supported_pt_extensions,
)


def get_available_models():
    return folder_paths.get_filename_list("AnimateDiff")


def get_model_path(model_name):
    return folder_paths.get_full_path("AnimateDiff", model_name)


def get_model_hash(file_path):
    with open(file_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        return hashlib.sha256(bytes).hexdigest()

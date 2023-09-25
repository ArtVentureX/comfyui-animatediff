import sys
import torch
import numpy as np
import subprocess
from PIL import Image


from .logger import logger

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def ensure_opencv():
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    try:
        import cv2
    except Exception as e:
        try:
            subprocess.check_call(pip_install + ['opencv-python'])
        except:
            logger.error(f"Failed to install 'opencv-python'. Please, install manually.")

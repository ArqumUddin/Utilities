"""
Image upscaling using Stable Diffusion X4 upscaler.
"""
import torch
import numpy as np
from typing import Optional, Union
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

from .utils import (
    auto_detect_device,
    load_image_as_pil,
    crop_pil_image_to_mask
)

def upscale_image_stable_diffusion(
    image: Union[str, np.ndarray, Image.Image],
    prompt: str,
    mask: Optional[Union[str, np.ndarray]] = None,
    crop_to_mask: bool = True,
    mask_expand_factor: float = 1.2,
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
    device: Optional[str] = None
) -> Image.Image:
    """
    Upscale an image using Stable Diffusion X4 upscaler.

    Args:
        image: Input image as file path, numpy array, or PIL Image
        prompt: Text prompt to guide upscaling (e.g., "Hand manipulates a bottle.")
        mask: Optional mask to crop image to object region (file path or numpy array)
        crop_to_mask: If True and mask provided, crop image to masked region before upscaling
        mask_expand_factor: Factor to expand bounding box around mask (default 1.2 = 20% larger)
        model_id: HuggingFace model identifier for upscaler
        device: Device for inference ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        PIL Image: Upscaled image (4x resolution)
    """
    device = auto_detect_device(device)

    print(f"Loading upscaler model: {model_id} on {device}")
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    pipeline = pipeline.to(device)
    low_res_img = load_image_as_pil(image)

    if mask is not None and crop_to_mask:
        low_res_img, bbox = crop_pil_image_to_mask(
            low_res_img,
            mask,
            expand_factor=mask_expand_factor
        )
        if bbox is not None:
            print(f"Cropped to bbox: {bbox}")

    print(f"Upscaling image with prompt: '{prompt}'")
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

    return upscaled_image

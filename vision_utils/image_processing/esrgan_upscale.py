"""
Image upscaling using Real-ESRGAN.
"""
import torch
import numpy as np
import cv2
from typing import Optional, Union
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from .utils import (
    auto_detect_device,
    load_image_as_cv2,
    crop_cv2_image_to_mask
)

def upscale_image_esrgan(
    image: Union[str, np.ndarray, Image.Image],
    mask: Optional[Union[str, np.ndarray]] = None,
    crop_to_mask: bool = True,
    mask_expand_factor: float = 1.2,
    model_name: str = "RealESRGAN_x4plus",
    model_path: Optional[str] = None,
    outscale: int = 4,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    device: Optional[str] = None
) -> Image.Image:
    """
    Upscale an image using Real-ESRGAN.

    Args:
        image: Input image as file path, numpy array, or PIL Image
        mask: Optional mask to crop image to object region (file path or numpy array)
        crop_to_mask: If True and mask provided, crop image to masked region before upscaling
        mask_expand_factor: Factor to expand bounding box around mask (default 1.2 = 20% larger)
        model_name: Model variant to use. Options:
            - "RealESRGAN_x4plus" (default): General 4x upscaling
            - "RealESRNet_x4plus": More conservative 4x upscaling
            - "RealESRGAN_x4plus_anime_6B": Optimized for anime images
            - "RealESRGAN_x2plus": 2x upscaling
        model_path: Optional custom path to model weights (.pth file)
        outscale: Output upscale factor (2 or 4)
        tile: Tile size for processing large images (0 = no tiling, 400-800 recommended for limited VRAM)
        tile_pad: Tile padding to avoid edge artifacts
        pre_pad: Pre-padding size
        device: Device for inference ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        PIL Image: Upscaled image
    """
    device = auto_detect_device(device)

    img = load_image_as_cv2(image)
    if mask is not None and crop_to_mask:
        img, bbox = crop_cv2_image_to_mask(
            img,
            mask,
            expand_factor=mask_expand_factor
        )
        if bbox is not None:
            print(f"Cropped to bbox: {bbox}")

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Choose from: RealESRGAN_x4plus, RealESRNet_x4plus, "
            f"RealESRGAN_x4plus_anime_6B, RealESRGAN_x2plus"
        )

    if model_path is None:
        model_path = model_url

    print(f"Loading Real-ESRGAN model: {model_name} on {device}")
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=(device == 'cuda'),  # Use FP16 for GPU
        device=device
    )
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print(f"Error during upscaling: {error}")
        if 'out of memory' in str(error):
            print("Try reducing tile size or using CPU device")
        raise

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output_pil = Image.fromarray(output_rgb)

    print(f"Upscaling complete: {img.shape[:2]} -> {output.shape[:2]}")

    return output_pil

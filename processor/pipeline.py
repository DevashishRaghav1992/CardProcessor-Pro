from .bg_removal import remove_background_and_crop
from .upscale import upscale_image


def run_card_pipeline(image_bytes: bytes):
    """
    Executes the full image processing pipeline.
    1. Background Removal & Tight Crop
    2. AI Upscaling
    Returns the final processed OpenCV BGR image.
    """
    # 1. Background removal and tight crop
    img = remove_background_and_crop(image_bytes)
    
    # 2. Upscale image
    img = upscale_image(img)
    
    return img

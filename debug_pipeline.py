import cv2
import sys
import os

from processor.bg_removal import remove_background_and_crop
from processor.upscale import upscale_image
from processor.zipper import standardize_and_zip

try:
    from processor.redaction import redact_names
    HAS_REDACTION = True
except ImportError:
    HAS_REDACTION = False

def run_debug_pipeline(image_path):
    print(f"Reading image from: {image_path}")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        
    # 1. Background removal and crop
    print("1. Running background removal...")
    cropped = remove_background_and_crop(img_bytes)
    cv2.imwrite("debug_1_cropped.jpg", cropped)
    
    # 2. Upscaling
    print("2. Running upscaling...")
    upscaled = upscale_image(cropped)
    cv2.imwrite("debug_2_upscaled.jpg", upscaled)
    
    # 3. Redaction (optional)
    if HAS_REDACTION:
        print("3. Running redaction...")
        final = redact_names(upscaled)
        cv2.imwrite("debug_3_redacted.jpg", final)
    else:
        print("3. Redaction skipped (easyocr not installed)")
        final = upscaled
    
    # 4. Standardize (Resize inside zipper logic)
    print("4. Standardizing and Zipping...")
    TARGET_WIDTH = 600
    TARGET_HEIGHT = 800
    resized = cv2.resize(final, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("debug_4_final_resized.jpg", resized)

    print("Pipeline Debug Complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_debug_pipeline(sys.argv[1])
    else:
        print("Please provide an image path.")

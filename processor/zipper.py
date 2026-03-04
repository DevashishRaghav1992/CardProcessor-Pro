import cv2
import zipfile
import io
import numpy as np
from typing import List

def standardize_and_zip(images: List[tuple[str, np.ndarray]]) -> io.BytesIO:
    """
    Takes a list of tuples (filename, image). Resizes images to standard size
    (e.g., 600x800), encodes them as PNG/JPG, and zips them.
    Returns an in-memory ZIP file stream.
    """
    TARGET_LONG_SIDE = 1200
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for idx, (filename, img) in enumerate(images):
            h, w = img.shape[:2]
            
            # Determine orientation to preserve aspect ratio
            if w > h:
                # Landscape
                new_w = TARGET_LONG_SIDE
                new_h = int(h * (TARGET_LONG_SIDE / w))
            else:
                # Portrait
                new_h = TARGET_LONG_SIDE
                new_w = int(w * (TARGET_LONG_SIDE / h))
                
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Encode as PNG to preserve transparent background
            success, buffer = cv2.imencode(".png", resized)
            if success:
                # Keep original name or rename
                name_without_ext = filename.rsplit(".", 1)[0]
                new_filename = f"{name_without_ext}_processed.png"
                
                zip_file.writestr(new_filename, buffer.tobytes())

    # Rewind buffer
    zip_buffer.seek(0)
    return zip_buffer

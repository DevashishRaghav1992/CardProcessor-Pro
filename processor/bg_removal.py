import cv2
import numpy as np
from rembg import remove


def remove_background_and_crop(image_bytes: bytes) -> np.ndarray:
    """
    Production-grade card isolation:
    1. Use rembg to remove background (produces alpha mask)
    2. Analyze contours in the alpha mask
    3. Keep ONLY the most card-like (rectangular) contour
    4. Discard all other objects (coins, clips, fingers, etc.)
    5. Return a tightly cropped BGRA image with clean transparent background
    
    Works for both vertical and horizontal card orientations.
    """
    # Step 1: Run rembg with alpha matting for clean edges
    bg_removed_bytes = remove(
        image_bytes,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    
    # Decode result (BGRA)
    nparr = np.frombuffer(bg_removed_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    if len(img.shape) != 3 or img.shape[2] != 4:
        # No alpha channel — return as is
        return img
    
    # Step 2: Extract alpha and find contours
    alpha = img[:, :, 3]
    _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Step 3: Score each contour for "card-likeness"
    # A card is: (a) the largest object, (b) very rectangular
    img_area = img.shape[0] * img.shape[1]
    best_contour = None
    best_score = -1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip tiny contours (noise)
        if area < img_area * 0.05:
            continue
        
        # Fit a minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        rect_area = cv2.contourArea(box)
        
        if rect_area == 0:
            continue
        
        # Rectangularity score: how well the contour fills its bounding rectangle
        # A perfect rectangle scores 1.0; a circle scores ~0.78
        rectangularity = area / rect_area
        
        # Check aspect ratio — standard credit card is 85.6mm x 53.98mm ≈ 1.586
        # Allow range from 1.3 to 1.8 (and its inverse for vertical cards)
        w_rect, h_rect = rect[1]
        if min(w_rect, h_rect) == 0:
            continue
        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
        
        # Card-like aspect ratio bonus
        if 1.2 <= aspect <= 1.9:
            aspect_score = 1.0
        else:
            aspect_score = 0.3  # Penalize non-card shapes
        
        # Combined score: weighted by area, rectangularity, and aspect ratio
        score = (area / img_area) * rectangularity * aspect_score
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    if best_contour is None:
        # Fallback: use the largest contour
        best_contour = max(contours, key=cv2.contourArea)
    
    # Step 4: Create a clean mask from ONLY the best contour
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(clean_mask, [best_contour], -1, 255, -1)
    
    # Step 5: Apply the clean mask to the alpha channel
    # This removes all non-card objects
    new_alpha = cv2.bitwise_and(alpha, clean_mask)
    img[:, :, 3] = new_alpha
    
    # Step 6: Tight crop around the card only
    _, crop_thresh = cv2.threshold(new_alpha, 10, 255, cv2.THRESH_BINARY)
    crop_contours, _ = cv2.findContours(crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if crop_contours:
        largest = max(crop_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Minimal padding (just 2px to avoid border clipping)
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        return img[y1:y2, x1:x2]
    
    return img

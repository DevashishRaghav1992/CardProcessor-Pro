import cv2
import numpy as np

# Lazy-loaded EasyOCR reader (only initialized when redact_names is called)
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def redact_names(img: np.ndarray) -> np.ndarray:
    """
    Scans the image for text. If it finds text, we apply cv2.inpaint
    to seamlessly remove the detected text, mimicking "redaction" without a flat background block.
    """
    # Create an initial mask of zeros
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    h, w = img.shape[:2]
    
    # Run OCR on the image
    results = _get_reader().readtext(img)
    
    has_text = False
    for (bbox, text, prob) in results:
        # Bounding box points: top_left, top_right, bottom_right, bottom_left
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        tr = (int(tr[0]), int(tr[1]))
        
        # Heuristic: Redact text only if it's in the typical name region
        # e.g., Bottom 35% of the card, Left 65% of the card
        if tl[1] > h * 0.65 and tr[0] < w * 0.65:
            has_text = True
            pad = 8  # Slightly wider pad for clean redaction
            cv2.rectangle(mask, (max(0, tl[0]-pad), max(0, tl[1]-pad)), (min(w, br[0]+pad), min(h, br[1]+pad)), 255, -1)

    if has_text:
        # Inpaint to remove the text seamlessly using the nearby background
        # Use cv2.INPAINT_TELEA or cv2.INPAINT_NS
        result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return result
        
    return img

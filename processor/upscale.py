import cv2
import numpy as np
import os
import urllib.request

# ─── Configuration ───────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
MODEL_FILENAME = "realesrgan_x4plus.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = (
    "https://huggingface.co/Qualcomm/Real-ESRGAN-x4plus/resolve/main/"
    "Real-ESRGAN-x4plus.onnx"
)
SCALE_FACTOR = 4
TILE_SIZE = 256       # Process in tiles to limit memory usage
TILE_OVERLAP = 16     # Overlap between tiles for seamless stitching

# Lazy-loaded ONNX session
_session = None


def _ensure_model():
    """Download the Real-ESRGAN ONNX model if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading Real-ESRGAN x4plus model to {MODEL_PATH} ...")
    print("(This is a one-time download, ~67 MB)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def _get_session():
    """Lazily initialize the ONNX Runtime inference session."""
    global _session
    if _session is None:
        import onnxruntime as ort

        ort.set_default_logger_severity(3)  # Suppress verbose logs
        _ensure_model()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _session


def _run_esrgan_tile(session, tile_bgr: np.ndarray) -> np.ndarray:
    """
    Run a single BGR tile through the Real-ESRGAN ONNX model.
    Input:  uint8 BGR HWC  →  Output: uint8 BGR HWC (4× larger)
    """
    # BGR → RGB, HWC → CHW, normalise to [0,1]
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)  # 1×3×H×W

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: tensor})[0][0]  # 3×(4H)×(4W)

    # CHW → HWC, clip, convert back to BGR uint8
    out_rgb = (result.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _upscale_tiled(session, img_bgr: np.ndarray) -> np.ndarray:
    """
    Upscale a full BGR image using tiled inference with overlap blending.
    This prevents OOM on large images while avoiding visible seams.
    """
    h, w = img_bgr.shape[:2]
    sf = SCALE_FACTOR

    # Pad image so dimensions are divisible by tile_size
    pad_h = (TILE_SIZE - h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - w % TILE_SIZE) % TILE_SIZE
    padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]

    # Output canvas
    out_h, out_w = ph * sf, pw * sf
    output = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float64)

    # Iterate over tiles with overlap
    step = TILE_SIZE - TILE_OVERLAP
    for y in range(0, ph, step):
        for x in range(0, pw, step):
            # Clamp tile boundaries
            ty = min(y, ph - TILE_SIZE)
            tx = min(x, pw - TILE_SIZE)
            tile = padded[ty : ty + TILE_SIZE, tx : tx + TILE_SIZE]

            # Run inference
            upscaled_tile = _run_esrgan_tile(session, tile)

            # Output coordinates
            oy, ox = ty * sf, tx * sf
            th, tw = upscaled_tile.shape[:2]

            # Accumulate with simple averaging (overlap regions get averaged)
            output[oy : oy + th, ox : ox + tw] += upscaled_tile.astype(np.float64)
            weight[oy : oy + th, ox : ox + tw] += 1.0

    # Average overlapping regions
    weight = np.maximum(weight, 1.0)
    output = (output / weight).clip(0, 255).astype(np.uint8)

    # Remove padding from output
    return output[: h * sf, : w * sf]


def upscale_image(img: np.ndarray) -> np.ndarray:
    """
    Upscale an image 4× using Real-ESRGAN via ONNX Runtime.
    Handles both BGR and BGRA (transparent) images.
    Falls back to local Lanczos upscaling if ONNX inference fails.
    """
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4

    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None

    try:
        session = _get_session()
        upscaled_bgr = _upscale_tiled(session, bgr)

        if alpha is not None:
            uh, uw = upscaled_bgr.shape[:2]
            upscaled_alpha = cv2.resize(alpha, (uw, uh), interpolation=cv2.INTER_LANCZOS4)
            _, upscaled_alpha = cv2.threshold(upscaled_alpha, 127, 255, cv2.THRESH_BINARY)
            return cv2.merge((
                upscaled_bgr[:, :, 0],
                upscaled_bgr[:, :, 1],
                upscaled_bgr[:, :, 2],
                upscaled_alpha,
            ))
        return upscaled_bgr

    except Exception as e:
        print(f"Real-ESRGAN upscale failed: {e}")
        print("Falling back to local Lanczos upscaling...")
        return _local_fallback_upscale(img)


def _local_fallback_upscale(img: np.ndarray) -> np.ndarray:
    """
    Fallback: local multi-pass Lanczos + sharpening if ONNX is unavailable.
    """
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4

    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None

    h, w = bgr.shape[:2]
    upscaled = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    upscaled = cv2.bilateralFilter(upscaled, d=5, sigmaColor=40, sigmaSpace=40)

    # Unsharp mask
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    upscaled = cv2.addWeighted(upscaled, 2.0, blurred, -1.0, 0)

    if alpha is not None:
        uh, uw = upscaled.shape[:2]
        upscaled_alpha = cv2.resize(alpha, (uw, uh), interpolation=cv2.INTER_LANCZOS4)
        _, upscaled_alpha = cv2.threshold(upscaled_alpha, 127, 255, cv2.THRESH_BINARY)
        return cv2.merge((upscaled[:, :, 0], upscaled[:, :, 1], upscaled[:, :, 2], upscaled_alpha))

    return upscaled

from __future__ import annotations
import numpy as np
import cv2

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    arr = np.asarray(img).astype(np.float32)
    # handle single-band 2D arrays
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    # handle singleton third axis e.g., (H, W, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # if more than 3 bands, take first 3
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    # normalize
    mn, mx = arr.min(), arr.max()
    if mx > 0 and mx != mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def vegetation_mask(img: np.ndarray, thresh: float = 20.0) -> np.ndarray:
    # ExG = 2*G - R - B
    R = img[..., 0].astype(np.float32)
    G = img[..., 1].astype(np.float32)
    B = img[..., 2].astype(np.float32)
    exg = 2 * G - R - B
    return (exg > thresh).astype(np.uint8)

def water_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # blue-ish hues
    lower = np.array([90, 30, 30])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return (mask > 0).astype(np.uint8)

def reliefs_mask(img: np.ndarray, thresh: float = 30.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mag = (mag / (mag.max() + 1e-9)) * 255.0
    return (mag > thresh).astype(np.uint8)

def cities_mask(img: np.ndarray, bright_thresh: int = 180) -> np.ndarray:
    # bright, low-green areas as proxy for built-up
    R = img[..., 0].astype(np.float32)
    G = img[..., 1].astype(np.float32)
    B = img[..., 2].astype(np.float32)
    brightness = (R + G + B) / 3.0
    mask = (brightness > bright_thresh) & (G < brightness * 0.7)
    return mask.astype(np.uint8)

def visualize_masks(img: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.5) -> np.ndarray:
    out = img.copy().astype(np.float32)
    colors = {
        'Vegetación': (34, 139, 34),
        'Agua': (30, 144, 255),
        'Relieves': (255, 165, 0),
        'Ciudades': (220, 20, 60),
    }
    for name, mask in masks.items():
        color = colors.get(name, (255, 255, 255))
        cm = np.zeros_like(out)
        for c in range(3):
            cm[..., c] = color[c]
        m = (mask > 0)[..., None]
        out = out * (1 - alpha * m) + cm * (alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def compute_stats(masks: dict[str, np.ndarray]) -> dict[str, float]:
    stats = {}
    for k, m in masks.items():
        total = m.size
        if total == 0:
            stats[k] = 0.0
        else:
            stats[k] = float(m.sum()) / float(total) * 100.0
    return stats

def cmap_to_rgb(cmap_name: str) -> np.ndarray:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    colors = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    return colors
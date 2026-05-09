"""
utils.py — Utility helpers: colour mapping, mask↔RGB conversion.
CS F407 Project Assignment II
"""

import numpy as np
import cv2
from config import CLASS_COLORS, NUM_CLASSES


def class_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D integer class-ID mask (H×W) to an RGB colour image (H×W×3).
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        rgb[mask == class_id] = color
    return rgb


def rgb_to_class_mask(rgb_mask: np.ndarray) -> np.ndarray:
    """
    Convert a TU Graz RGB annotation mask (H×W×3) to a 2-D integer class-ID
    mask (H×W).  Pixels not matching any known colour receive label 0.
    """
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    for class_id, color in enumerate(CLASS_COLORS):
        match = np.all(rgb_mask == np.array(color, dtype=np.uint8), axis=-1)
        class_mask[match] = class_id
    return class_mask


def apply_colormap(depth_uint8: np.ndarray) -> np.ndarray:
    """
    Apply JET colormap to a uint8 depth map for validation visualisation.
    Returns a BGR image (OpenCV convention).
    """
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)


def draw_rotated_box(img: np.ndarray, cx: int, cy: int,
                     w: int, h: int, angle: float,
                     color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draw a rotated bounding box on *img* (in-place copy).

    Parameters
    ----------
    img    : BGR image (H×W×3)
    cx, cy : centre of the box
    w, h   : width and height of the box
    angle  : rotation in degrees
    color  : BGR colour tuple
    """
    out = img.copy()
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle))
    box  = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(out, [box], 0, color, thickness)
    return out


def put_text_with_background(img: np.ndarray, text: str,
                              origin, font_scale=0.45,
                              text_color=(255, 255, 255),
                              bg_color=(0, 0, 0)) -> np.ndarray:
    """
    Draw *text* at *origin* with a filled background rectangle.
    Returns a copy of the image.
    """
    out  = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, 1)
    x, y = origin
    cv2.rectangle(out, (x, y - th - bl - 2), (x + tw + 2, y + bl), bg_color, -1)
    cv2.putText(out, text, (x, y), font, font_scale, text_color, 1, cv2.LINE_AA)
    return out


def resize_keep_scale(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize *img* to (target_w × target_h) using bilinear interpolation."""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_mask_nearest(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a class-ID mask to (target_w × target_h) using nearest-neighbour
    interpolation to avoid introducing spurious class IDs."""
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h),
                      interpolation=cv2.INTER_NEAREST).astype(np.int64)

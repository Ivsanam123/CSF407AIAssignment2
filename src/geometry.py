"""
geometry.py — Safe-zone mask generation, candidate search, cost computation.
CS F407 Project Assignment II
"""

import math
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from config import (SAFE_CLASSES, MAX_SLOPE, BOX_W, BOX_H,
                    SEARCH_STEP, ANGLE_STEP, INTERIOR_SAMPLE_STEP,
                    MAIN_W, MAIN_H, ANN_W, ANN_H,
                    WEIGHT_DISTANCE, WEIGHT_ROUGHNESS, WEIGHT_SEMANTIC,
                    DEVICE, MIDAS_MODEL)
from semantic_brain import get_semantic_penalty, sample_interior_points
from utils import resize_mask_nearest


# ─────────────────────────────────────────────────────────────────────────────
# Depth estimation (MiDaS)
# ─────────────────────────────────────────────────────────────────────────────

_midas_model   = None
_midas_transform = None


def _load_midas():
    global _midas_model, _midas_transform
    if _midas_model is None:
        _midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL,
                                       pretrained=True)
        _midas_model.to(DEVICE).eval()
        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms")
        _midas_transform = transforms_hub.small_transform


def estimate_depth(img_rgb: np.ndarray) -> np.ndarray:
    """
    Run MiDaS on *img_rgb* (H×W×3 uint8, 800×600).

    Returns a uint8 depth map (800×600) where:
        bright (255) = close to drone
        dark   (0)   = far from drone
    """
    _load_midas()

    input_batch = _midas_transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        depth = _midas_model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(MAIN_H, MAIN_W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_np = depth.cpu().numpy()
    # Normalise to [0, 255] — higher value = closer to drone
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return depth_uint8


# ─────────────────────────────────────────────────────────────────────────────
# Safe-zone binary mask
# ─────────────────────────────────────────────────────────────────────────────

def build_safe_mask(seg_mask: np.ndarray) -> np.ndarray:
    """
    Create a binary safety mask from the segmentation map.

    Safe classes (1, 3, 4) → 255 (white)
    All others              → 0   (black)

    Parameters
    ----------
    seg_mask : 2-D int64 array (H×W) at 800×600 resolution

    Returns
    -------
    uint8 binary mask (H×W)
    """
    safe_mask = np.zeros(seg_mask.shape, dtype=np.uint8)
    for cls in SAFE_CLASSES:
        safe_mask[seg_mask == cls] = 255
    return safe_mask


# ─────────────────────────────────────────────────────────────────────────────
# Slope (roughness) calculation
# ─────────────────────────────────────────────────────────────────────────────

def _sobel_gradient(depth_map: np.ndarray) -> np.ndarray:
    """Compute per-pixel Sobel gradient magnitude from a depth map."""
    depth_f = depth_map.astype(np.float32) / 255.0
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def mean_slope(depth_map: np.ndarray,
               cx: int, cy: int,
               w: int, h: int,
               angle_deg: float) -> float:
    """
    Compute mean Sobel gradient magnitude inside a candidate box.
    """
    grad = _sobel_gradient(depth_map)
    img_h, img_w = grad.shape
    points = sample_interior_points(cx, cy, w, h, angle_deg)

    valid = ((points[:, 0] >= 0) & (points[:, 0] < img_w) &
             (points[:, 1] >= 0) & (points[:, 1] < img_h))
    points = points[valid]

    if len(points) == 0:
        return float("inf")

    return float(grad[points[:, 1], points[:, 0]].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Candidate search (rotational grid)
# ─────────────────────────────────────────────────────────────────────────────

def _box_interior_all_white(safe_mask: np.ndarray,
                             cx: int, cy: int,
                             w: int, h: int,
                             angle_deg: float) -> bool:
    """Return True iff all sampled interior points fall on white (safe) pixels."""
    img_h, img_w = safe_mask.shape
    points = sample_interior_points(cx, cy, w, h, angle_deg)

    valid = ((points[:, 0] >= 0) & (points[:, 0] < img_w) &
             (points[:, 1] >= 0) & (points[:, 1] < img_h))

    if valid.sum() == 0:
        return False

    points = points[valid]
    return bool(np.all(safe_mask[points[:, 1], points[:, 0]] == 255))


def _box_within_bounds(cx: int, cy: int, w: int, h: int) -> bool:
    """
    Conservative axis-aligned bounds check — fast pre-filter before the
    more expensive rotated-interior check.
    """
    half_diag = math.ceil(math.sqrt(w**2 + h**2) / 2)
    return (cx - half_diag >= 0 and cx + half_diag < MAIN_W and
            cy - half_diag >= 0 and cy + half_diag < MAIN_H)


def find_candidates(safe_mask: np.ndarray,
                    depth_map: np.ndarray,
                    box_w: int = BOX_W,
                    box_h: int = BOX_H) -> list:
    """
    Rotational grid search for valid candidate landing boxes.

    A candidate is valid iff:
      - It fits within image bounds.
      - All sampled interior points are white in safe_mask.
      - Mean Sobel gradient < MAX_SLOPE.

    Returns
    -------
    List of dicts: {"cx", "cy", "angle", "roughness"}
    """
    candidates = []
    angles = range(0, 180, ANGLE_STEP)

    for cy in range(box_h, MAIN_H - box_h, SEARCH_STEP):
        for cx in range(box_w, MAIN_W - box_w, SEARCH_STEP):
            # Fast bounds pre-filter
            if not _box_within_bounds(cx, cy, box_w, box_h):
                continue

            for angle in angles:
                if not _box_interior_all_white(safe_mask, cx, cy,
                                                box_w, box_h, angle):
                    continue

                slope = mean_slope(depth_map, cx, cy, box_w, box_h, angle)
                if slope >= MAX_SLOPE:
                    continue

                candidates.append({
                    "cx":        cx,
                    "cy":        cy,
                    "angle":     angle,
                    "roughness": slope / MAX_SLOPE,   # normalised to [0,1)
                })

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Cost function + ranking
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost(candidate: dict,
                 target_x: int, target_y: int,
                 seg_mask: np.ndarray,
                 active_nodes: list) -> float:
    """
    Scalar cost for a single candidate.

    Cost = 0.4 × distance + 0.2 × roughness + 0.4 × semantic_penalty
    """
    cx, cy    = candidate["cx"], candidate["cy"]
    roughness = candidate["roughness"]

    distance = math.sqrt((cx - target_x)**2 + (cy - target_y)**2) / 1000.0

    semantic_penalty = get_semantic_penalty(
        seg_mask, cx, cy, BOX_W, BOX_H, candidate["angle"], active_nodes)

    total = (WEIGHT_DISTANCE  * distance
           + WEIGHT_ROUGHNESS * roughness
           + WEIGHT_SEMANTIC  * semantic_penalty)

    candidate["distance"]         = distance
    candidate["semantic_penalty"] = semantic_penalty
    candidate["cost"]             = total
    return total


def rank_candidates(candidates: list,
                    target_x: int, target_y: int,
                    seg_mask: np.ndarray,
                    active_nodes: list) -> list:
    """
    Compute cost for every candidate and return sorted list (lowest cost first).
    """
    for c in candidates:
        compute_cost(c, target_x, target_y, seg_mask, active_nodes)
    return sorted(candidates, key=lambda c: c["cost"])

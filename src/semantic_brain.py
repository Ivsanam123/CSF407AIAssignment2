"""
semantic_brain.py — Terrain classifier and mission config parser.
CS F407 Project Assignment II

Responsibilities
----------------
1. Given a candidate bounding box and a segmentation mask, determine the
   dominant terrain type by sampling interior pixels.
2. Parse mission_config.json at runtime to extract active package traits
   (active L1 source nodes for the knowledge graph).
"""

import json
import math
import numpy as np

from config import CLASS_TO_TERRAIN, INTERIOR_SAMPLE_STEP
from knowledge_graph import think


# ── Trait → source node mapping ───────────────────────────────────────────────
# Maps mission_config.json package property names → KG Layer-1 node names
TRAIT_TO_NODE = {
    "fragile":   "Fragile",
    "valuable":  "Valuable",
    "biohazard": "Biohazard",
    "heavy":     "Heavy",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mission config parser
# ─────────────────────────────────────────────────────────────────────────────

def load_active_nodes(config_path: str = "mission_config.json") -> list:
    """
    Parse *config_path* (JSON) and return the list of active KG Layer-1
    source node names based on the package properties that evaluate to True.

    Example JSON
    ------------
    {
        "mission_id": "OP-DELTA-9",
        "package": {
            "type": "medical_vials",
            "heavy":     true,
            "fragile":   false,
            "valuable":  false,
            "biohazard": false
        }
    }
    Returns: ["Heavy"]
    """
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    package = cfg.get("package", {})
    active  = [TRAIT_TO_NODE[trait]
               for trait in TRAIT_TO_NODE
               if package.get(trait, False)]
    return active


# ─────────────────────────────────────────────────────────────────────────────
# 2. Interior point sampler
# ─────────────────────────────────────────────────────────────────────────────

def _get_rotated_box_points(cx: float, cy: float,
                             w: float, h: float,
                             angle_deg: float) -> np.ndarray:
    """Return the four corner points of a rotated rectangle."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0

    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
    rotated = corners @ rot.T + np.array([cx, cy])
    return rotated.astype(np.float32)


def sample_interior_points(cx: int, cy: int, w: int, h: int,
                            angle_deg: float,
                            step: int = INTERIOR_SAMPLE_STEP) -> np.ndarray:
    """
    Sample a regular grid of pixel coordinates inside a rotated bounding box.

    Returns
    -------
    np.ndarray of shape (N, 2) with (col, row) = (x, y) coordinates.
    """
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    xs = np.arange(-w // 2 + step, w // 2, step)
    ys = np.arange(-h // 2 + step, h // 2, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    local = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(float)

    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
    world = local @ rot.T + np.array([cx, cy])
    return world.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dominant terrain classifier
# ─────────────────────────────────────────────────────────────────────────────

def dominant_terrain(seg_mask: np.ndarray,
                     cx: int, cy: int,
                     w: int, h: int,
                     angle_deg: float) -> str:
    """
    Identify the dominant terrain type inside the candidate bounding box.

    Parameters
    ----------
    seg_mask  : 2-D int64 array (H×W) of class IDs at 800×600 resolution
    cx, cy    : centre of the candidate box (in 800×600 pixel space)
    w, h      : box dimensions
    angle_deg : rotation angle

    Returns
    -------
    One of "Pavement", "Grass", "Dirt" (or "Unknown" if no safe class found).
    """
    img_h, img_w = seg_mask.shape
    points = sample_interior_points(cx, cy, w, h, angle_deg)

    # Filter points that are inside image bounds
    valid = ((points[:, 0] >= 0) & (points[:, 0] < img_w) &
             (points[:, 1] >= 0) & (points[:, 1] < img_h))
    points = points[valid]

    if len(points) == 0:
        return "Unknown"

    # Gather class IDs at sampled points
    class_ids = seg_mask[points[:, 1], points[:, 0]]

    # Count frequency of each safe class
    counts = {}
    for cid in class_ids:
        if cid in CLASS_TO_TERRAIN:
            counts[cid] = counts.get(cid, 0) + 1

    if not counts:
        return "Unknown"

    dominant_class = max(counts, key=counts.get)
    return CLASS_TO_TERRAIN[dominant_class]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Semantic penalty retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_semantic_penalty(seg_mask: np.ndarray,
                         cx: int, cy: int,
                         w: int, h: int,
                         angle_deg: float,
                         active_nodes: list) -> float:
    """
    Full semantic scoring pipeline for one candidate box:
      1. Find dominant terrain via segmentation sampling.
      2. Look up penalty from the knowledge graph.

    Returns penalty in [0.0, 1.0]; returns 1.0 for "Unknown" terrain.
    """
    terrain = dominant_terrain(seg_mask, cx, cy, w, h, angle_deg)
    if terrain == "Unknown":
        return 1.0   # maximum penalty — unknown terrain is not safe

    penalties = think(active_nodes)
    return penalties.get(terrain, 1.0)

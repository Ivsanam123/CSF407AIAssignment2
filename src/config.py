"""
config.py — Centralized configuration for all constants.
CS F407 Project Assignment II
"""

import torch

# ── Image sizes ──────────────────────────────────────────────────────────────
MAIN_W, MAIN_H       = 800, 600   # working resolution for all geometric ops
ANN_W,  ANN_H        = 256, 256   # ANN input resolution

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 23   # labels 0-22

# TU Graz Semantic Drone Dataset — 23 class names (index = class id)
CLASS_NAMES = [
    "unlabeled",        # 0
    "paved-area",       # 1  ← SAFE
    "dirt",             # 2  (not safe by default)
    "grass",            # 3  ← SAFE
    "gravel",           # 4  ← SAFE
    "water",            # 5
    "rocks",            # 6
    "pool",             # 7
    "vegetation",       # 8
    "roof",             # 9
    "wall",             # 10
    "window",           # 11
    "door",             # 12
    "fence",            # 13
    "fence-pole",       # 14
    "person",           # 15
    "dog",              # 16
    "car",              # 17
    "bicycle",          # 18
    "tree",             # 19
    "bald-tree",        # 20
    "ar-marker",        # 21
    "obstacle",         # 22
]

# RGB colours for each class (used in utils.py for visualisation)
CLASS_COLORS = [
    (0,   0,   0),    # 0  unlabeled
    (128, 64,  128),  # 1  paved-area
    (130, 76,  0),    # 2  dirt
    (0,   102, 0),    # 3  grass
    (112, 103, 87),   # 4  gravel
    (28,  42,  168),  # 5  water
    (48,  41,  30),   # 6  rocks
    (0,   50,  89),   # 7  pool
    (107, 142, 35),   # 8  vegetation
    (70,  70,  70),   # 9  roof
    (102, 102, 156),  # 10 wall
    (254, 228, 12),   # 11 window
    (254, 148, 12),   # 12 door
    (190, 153, 153),  # 13 fence
    (153, 153, 153),  # 14 fence-pole
    (255, 22,  96),   # 15 person
    (102, 51,  0),    # 16 dog
    (9,   143, 150),  # 17 car
    (119, 11,  32),   # 18 bicycle
    (51,  51,  0),    # 19 tree
    (190, 250, 190),  # 20 bald-tree
    (112, 150, 146),  # 21 ar-marker
    (2,   135, 115),  # 22 obstacle
]

# ── Safe landing classes ───────────────────────────────────────────────────────
SAFE_CLASSES = [1, 3, 4]   # paved-area, grass, gravel

# Mapping class id → terrain type for knowledge graph lookup
CLASS_TO_TERRAIN = {
    1: "Pavement",   # paved-area
    3: "Grass",      # grass
    4: "Dirt",       # gravel (treated as Dirt for KG purposes)
}

# ── Cost function weights ──────────────────────────────────────────────────────
WEIGHT_DISTANCE  = 0.4
WEIGHT_ROUGHNESS = 0.2
WEIGHT_SEMANTIC  = 0.4

# ── Depth / slope ──────────────────────────────────────────────────────────────
MAX_SLOPE = 0.15   # candidates with mean Sobel >= this are discarded

# ── Bounding box search ────────────────────────────────────────────────────────
BOX_W       = 80    # candidate landing box width  (at 800x600 scale)
BOX_H       = 80    # candidate landing box height
SEARCH_STEP = 20    # grid stride in pixels
ANGLE_STEP  = 10    # rotation step in degrees (0–170)
INTERIOR_SAMPLE_STEP = 8   # stride for sampling interior pixels

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS = 20
LEARNING_RATE   = 1e-3
TRAIN_SPLIT     = 0.8     # fraction used for training (rest = validation)
CHECKPOINT_PATH = "/content/drive/MyDrive/CS_F407_Project/best_model.pth"

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── MiDaS ──────────────────────────────────────────────────────────────────────
MIDAS_MODEL = "MiDaS_small"

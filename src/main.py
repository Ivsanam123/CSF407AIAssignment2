"""
main.py — CLI entry point and full inference pipeline.
CS F407 Project Assignment II

Usage:
    python main.py --image path/to/image.jpg --target_x 700 --target_y 200
                   [--config mission_config.json]
                   [--checkpoint best_model.pth]
                   [--output_dir .]
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from config import (MAIN_W, MAIN_H, ANN_W, ANN_H, DEVICE,
                    CHECKPOINT_PATH, BOX_W, BOX_H)
from model import build_model
from geometry import (estimate_depth, build_safe_mask,
                      find_candidates, rank_candidates)
from semantic_brain import load_active_nodes, dominant_terrain
from knowledge_graph import think
from utils import (class_mask_to_rgb, apply_colormap,
                   draw_rotated_box, put_text_with_background,
                   resize_mask_nearest)


# ── Image pre-processing transform for ANN ────────────────────────────────────
_IMG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_image(path: str):
    """Load image, convert to RGB, resize to 800×600."""
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"[ERROR] Cannot open image: {path}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (MAIN_W, MAIN_H), interpolation=cv2.INTER_LINEAR)
    return rgb


def run_segmentation(model, img_rgb: np.ndarray) -> np.ndarray:
    """
    ANN inference:
      1. Downscale to 256×256.
      2. Run model.
      3. Upscale output mask back to 800×600.

    Returns int64 class-ID mask at 800×600.
    """
    # Downscale
    img_ann = cv2.resize(img_rgb, (ANN_W, ANN_H), interpolation=cv2.INTER_LINEAR)
    tensor  = _IMG_TRANSFORM(img_ann).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)                      # (1, C, 256, 256)
        pred   = logits.argmax(dim=1).squeeze(0)    # (256, 256)
        pred_np = pred.cpu().numpy().astype(np.int64)

    # Upscale back to 800×600
    seg_mask = resize_mask_nearest(pred_np, MAIN_W, MAIN_H)
    return seg_mask


def draw_output(img_rgb: np.ndarray,
                best: dict,
                target_x: int,
                target_y: int,
                active_nodes: list) -> np.ndarray:
    """Draw result bounding box, target marker and cost annotation."""
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Green oriented bounding box
    out_bgr = draw_rotated_box(out_bgr,
                               best["cx"], best["cy"],
                               BOX_W, BOX_H, best["angle"],
                               color=(0, 255, 0), thickness=2)

    # Red star / circle at target location
    cv2.drawMarker(out_bgr, (target_x, target_y), (0, 0, 255),
                   cv2.MARKER_STAR, 18, 2)

    # Annotation text
    label = (f"Best: ({best['cx']},{best['cy']}) "
             f"@ {best['angle']}deg | Cost: {best['cost']:.4f}")
    out_bgr = put_text_with_background(out_bgr, label, (6, 18))

    return out_bgr


def draw_analysis_dashboard(img_rgb: np.ndarray,
                              seg_mask: np.ndarray,
                              depth_uint8: np.ndarray,
                              safe_mask: np.ndarray,
                              best_img: np.ndarray,
                              best: dict,
                              active_nodes: list) -> np.ndarray:
    """
    Compose the 6-panel output_analysis.jpg dashboard:
    [Original | Segmentation | Depth Map]
    [Safe Mask | Best Placement | Cost Breakdown]
    """
    H, W = MAIN_H, MAIN_W

    # Panel 1 — Original
    p1 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Panel 2 — Semantic segmentation
    seg_rgb = class_mask_to_rgb(seg_mask)
    p2 = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    # Panel 3 — Depth map (colormap)
    p3 = apply_colormap(depth_uint8)   # already BGR

    # Panel 4 — Safe zone mask (3-channel)
    p4 = cv2.cvtColor(safe_mask, cv2.COLOR_GRAY2BGR)

    # Panel 5 — Best placement
    p5 = best_img.copy()

    # Panel 6 — Cost breakdown text panel
    p6 = np.ones((H, W, 3), dtype=np.uint8) * 245
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    color     = (30, 30, 30)
    lh        = 22

    lines = [
        "BEST PLACEMENT (SEMANTIC REASONING)",
        "",
        f"Position: ({best['cx']}, {best['cy']})",
        f"Angle: {best['angle']}deg",
        f"Terrain: {best.get('terrain', 'N/A')}",
        f"Distance: {best['distance']*1000:.2f}px",
        f"Roughness: {best['roughness']:.4f}",
        f"Semantic Penalty: {best['semantic_penalty']:.4f}",
        "",
        "Mission Factors:",
        ", ".join(active_nodes) if active_nodes else "None",
        "",
        "Cost Breakdown:",
        f"Distance  (0.4): {0.4 * best['distance']:.4f}",
        f"Roughness (0.2): {0.2 * best['roughness']:.4f}",
        f"Semantic  (0.4): {0.4 * best['semantic_penalty']:.4f}",
        f"TOTAL:           {best['cost']:.4f}",
    ]

    for i, line in enumerate(lines):
        y = 30 + i * lh
        fw = 600 if i == 0 else 400
        cv2.putText(p6, line, (10, y), font, font_scale, color, 1, cv2.LINE_AA)

    def add_title(panel, title):
        return put_text_with_background(panel, title, (4, 14),
                                         font_scale=0.38,
                                         text_color=(255, 255, 255),
                                         bg_color=(50, 50, 50))

    p1 = add_title(p1, f"Original Image ({W}x{H})")
    p2 = add_title(p2, "Semantic Segmentation (U-Net)")
    p3 = add_title(p3, "Depth Map (MiDaS)")
    p4 = add_title(p4, "Safe Zone Mask")
    p5 = add_title(p5, f"Best Placement (Cost: {best['cost']:.4f})")
    p6 = add_title(p6, "BEST PLACEMENT (SEMANTIC REASONING)")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    dashboard = np.vstack([row1, row2])
    # Resize dashboard to a reasonable output resolution
    dh, dw = dashboard.shape[:2]
    dashboard = cv2.resize(dashboard, (min(dw, 2400), min(dh, 800)),
                            interpolation=cv2.INTER_AREA)
    return dashboard


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous drone landing zone finder (CS F407 Project II)")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--target_x",  type=int, required=True,
                        help="Target X coordinate (0-800)")
    parser.add_argument("--target_y",  type=int, required=True,
                        help="Target Y coordinate (0-600)")
    parser.add_argument("--config",     default="mission_config.json",
                        help="Path to mission_config.json")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to best_model.pth")
    parser.add_argument("--output_dir", default=".",
                        help="Directory to save output images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[main] Device: {DEVICE}")
    print(f"[main] Image:  {args.image}")
    print(f"[main] Target: ({args.target_x}, {args.target_y})")

    # ── Step 1: Load image ────────────────────────────────────────────────────
    print("[1/6] Loading image...")
    img_rgb = load_image(args.image)

    # ── Step 2: Load model & run segmentation ─────────────────────────────────
    print("[2/6] Running semantic segmentation...")
    model = build_model(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    seg_mask = run_segmentation(model, img_rgb)

    # ── Step 3: Depth estimation ──────────────────────────────────────────────
    print("[3/6] Estimating depth with MiDaS...")
    depth_uint8 = estimate_depth(img_rgb)

    # ── Step 4: Build safe mask & find candidates ─────────────────────────────
    print("[4/6] Building safe-zone mask and searching candidates...")
    safe_mask  = build_safe_mask(seg_mask)
    candidates = find_candidates(safe_mask, depth_uint8)
    print(f"      Found {len(candidates)} valid candidate(s).")

    if not candidates:
        print("[WARN] No safe landing zone found in this image.")
        sys.exit(0)

    # ── Step 5: Load mission config & rank by cost ────────────────────────────
    print("[5/6] Loading mission config & ranking candidates...")
    active_nodes = load_active_nodes(args.config)
    print(f"      Active nodes: {active_nodes}")
    ranked = rank_candidates(candidates, args.target_x, args.target_y,
                             seg_mask, active_nodes)
    best = ranked[0]

    # Attach terrain label
    best["terrain"] = dominant_terrain(seg_mask,
                                        best["cx"], best["cy"],
                                        BOX_W, BOX_H, best["angle"])

    print(f"\n[RESULT]")
    print(f"  Position       : ({best['cx']}, {best['cy']})")
    print(f"  Angle          : {best['angle']} deg")
    print(f"  Terrain        : {best['terrain']}")
    print(f"  Distance cost  : {0.4 * best['distance']:.4f}")
    print(f"  Roughness cost : {0.2 * best['roughness']:.4f}")
    print(f"  Semantic cost  : {0.4 * best['semantic_penalty']:.4f}")
    print(f"  TOTAL cost     : {best['cost']:.4f}")

    # ── Step 6: Render & save outputs ─────────────────────────────────────────
    print("[6/6] Rendering output images...")
    out_img = draw_output(img_rgb, best, args.target_x, args.target_y,
                          active_nodes)

    dashboard = draw_analysis_dashboard(
        img_rgb, seg_mask, depth_uint8, safe_mask, out_img, best, active_nodes)

    out_path       = os.path.join(args.output_dir, "output.jpg")
    analysis_path  = os.path.join(args.output_dir, "output_analysis.jpg")

    cv2.imwrite(out_path, out_img)
    cv2.imwrite(analysis_path, dashboard)

    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {analysis_path}")
    print("\n[main] Done.")


if __name__ == "__main__":
    main()

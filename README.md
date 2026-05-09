# CS F407 — Project Assignment II
## Autonomous Drone Landing Zone AI

---

## Setup

```bash
conda env create -f config.yml
conda activate CSF407_2026_ID1
```

---

## Training the ANN

1. Download the TU Graz Semantic Drone Dataset (`semantics_drone_dataset_semantics_v1.1.zip`) from the official site.
2. Extract so the folder structure is:
   ```
   dataset/semantic_drone_dataset/
       original_images/
       label_images_semantic/
   ```
3. Run training (Kaggle/Colab T4 GPU recommended):

```bash
cd src
python train.py --data_root /path/to/extracted/dataset
```

The best checkpoint is saved automatically to `src/best_model.pth`.

---

## Running Inference

```bash
cd src
python main.py \
    --image path/to/aerial_image.jpg \
    --target_x 700 \
    --target_y 200 \
    --config mission_config.json \
    --checkpoint best_model.pth \
    --output_dir ./results
```

### Output files

| File | Description |
|------|-------------|
| `output.jpg` | Input image with optimal green bounding box + red target star |
| `output_analysis.jpg` | 6-panel dashboard: original, segmentation, depth, safe mask, best placement, cost breakdown |

---

## Pipeline Overview

```
Input image (any resolution)
    ↓  resize to 800×600
Semantic Segmentation (U-Net, 256×256 → upscale back to 800×600)
    ↓
Depth Estimation (MiDaS_small)
    ↓
Binary Safe Mask (classes 1, 3, 4 → white)
    ↓
Rotational Grid Search → valid candidates (slope < 0.15)
    ↓
Knowledge Graph Penalty (3-layer spreading activation, no if/else)
    ↓
Cost = 0.4×distance + 0.2×roughness + 0.4×semantic_penalty
    ↓
Best candidate → green oriented bounding box on output image
```

---

## Module Structure

| File | Responsibility |
|------|---------------|
| `config.py` | All constants (image sizes, class labels, cost weights, device) |
| `dataset.py` | `GrazDataset` class: load images + RGB masks → CHW tensors |
| `model.py` | U-Net CNN: 4 encoder + 4 decoder layers |
| `train.py` | Training loop, Cross-Entropy loss, Adam, saves best checkpoint |
| `geometry.py` | Safe mask, MiDaS depth, rotational grid search, cost ranking |
| `knowledge_graph.py` | 3-layer spreading activation graph, `think()` function |
| `semantic_brain.py` | Terrain classifier + `mission_config.json` parser |
| `utils.py` | Colour mapping, mask↔RGB, drawing helpers |
| `main.py` | CLI entry point, orchestrates full inference pipeline |

---

## Knowledge Graph Design

**Layer 1 (source):** Fragile · Valuable · Biohazard · Heavy  
**Layer 2 (property):** Hard · Wet · Slippery · Dirty · Visible · Contaminated · Soft · Unstable  
**Layer 3 (terrain):** Pavement · Grass · Dirt  

All penalty scores are derived purely from edge-weight accumulation — no `if/else` chains.

**Example:** Active nodes = [Fragile, Valuable]  
→ Grass receives the lowest penalty (soft landing preferred for fragile cargo)

---

## Results


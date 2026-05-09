"""
train.py — Training loop for the semantic segmentation ANN.
CS F407 Project Assignment II

Run:
    python train.py --data_root /path/to/dataset_root
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
                    TRAIN_SPLIT, CHECKPOINT_PATH, DEVICE, NUM_CLASSES)
from dataset import GrazDataset
from model import build_model


def train(data_root: str):
    print(f"[train] device = {DEVICE}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = GrazDataset(data_root, split="train", train_frac=TRAIN_SPLIT)
    val_ds   = GrazDataset(data_root, split="val",   train_frac=TRAIN_SPLIT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[train] train={len(train_ds)} images, val={len(val_ds)} images")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = build_model(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimiser.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits  = model(imgs)
                loss    = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)

                preds   = logits.argmax(dim=1)
                correct += (preds == masks).sum().item()
                total   += masks.numel()
        val_loss /= len(val_ds)
        val_acc   = 100.0 * correct / total

        scheduler.step(val_loss)

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.2f}%", end="")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ← saved best_model.pth", end="")
        print()

    print(f"\n[train] Done. Best val_loss = {best_val_loss:.4f}")
    print(f"[train] Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to the extracted dataset root folder")
    args = parser.parse_args()
    train(args.data_root)

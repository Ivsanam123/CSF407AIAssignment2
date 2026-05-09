"""
model.py — U-Net style CNN with exactly 4 encoder + 4 decoder layers.
CS F407 Project Assignment II
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Two consecutive Conv→BN→ReLU operations (standard U-Net building block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegNet(nn.Module):
    """
    U-Net with exactly 4 encoder layers and 4 decoder layers.

    Encoder:  3 → 64 → 128 → 256 → 512  (with MaxPool between stages)
    Bottleneck: 512 → 1024
    Decoder: 1024 → 512 → 256 → 128 → 64  (with skip connections)
    Head:    64 → NUM_CLASSES (1×1 conv)

    Input shape  : (B, 3, 256, 256)
    Output shape : (B, NUM_CLASSES, 256, 256)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Encoder (4 layers) ───────────────────────────────────────────────
        self.enc1 = ConvBlock(3,   64)    # 256×256
        self.enc2 = ConvBlock(64,  128)   # 128×128
        self.enc3 = ConvBlock(128, 256)   # 64×64
        self.enc4 = ConvBlock(256, 512)   # 32×32

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(512, 1024)   # 16×16

        # ── Decoder (4 layers) ───────────────────────────────────────────────
        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4  = ConvBlock(1024, 512)   # skip from enc4

        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3  = ConvBlock(512, 256)    # skip from enc3

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2  = ConvBlock(256, 128)    # skip from enc2

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1  = ConvBlock(128, 64)     # skip from enc1

        # ── Output head ──────────────────────────────────────────────────────
        self.head  = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # (B, 64,  256, 256)
        e2 = self.enc2(self.pool(e1))   # (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))   # (B, 256, 64,  64)
        e4 = self.enc4(self.pool(e3))   # (B, 512, 32,  32)

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))   # (B, 1024, 16, 16)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)   # (B, NUM_CLASSES, 256, 256)


def build_model(device: str = "cpu") -> SegNet:
    """Instantiate and move model to *device*."""
    model = SegNet(num_classes=NUM_CLASSES).to(device)
    return model

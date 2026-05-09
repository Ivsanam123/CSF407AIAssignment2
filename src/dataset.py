
import os, glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import MAIN_W, MAIN_H, ANN_W, ANN_H
from utils import rgb_to_class_mask

class GrazDataset(Dataset):
    def __init__(self, root: str, split: str = "train", train_frac: float = 0.8):
        img_dir  = os.path.join(root, "training_set", "images")
        mask_dir = os.path.join(root, "training_set", "gt", "semantic", "label_images")

        all_imgs = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.JPG")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )

        if len(all_imgs) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        n_train = int(len(all_imgs) * train_frac)
        self.img_paths = all_imgs[:n_train] if split == "train" else all_imgs[n_train:]
        self.mask_dir  = mask_dir

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_main = cv2.resize(img_rgb,  (MAIN_W, MAIN_H), interpolation=cv2.INTER_LINEAR)
        img_ann  = cv2.resize(img_main, (ANN_W,  ANN_H),  interpolation=cv2.INTER_LINEAR)

        base      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        mask_bgr  = cv2.imread(mask_path)
        if mask_bgr is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask_rgb  = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask_main = cv2.resize(mask_rgb,  (MAIN_W, MAIN_H), interpolation=cv2.INTER_NEAREST)
        mask_ann  = cv2.resize(mask_main, (ANN_W,  ANN_H),  interpolation=cv2.INTER_NEAREST)

        class_mask  = rgb_to_class_mask(mask_ann)
        img_tensor  = self.img_transform(img_ann)
        mask_tensor = torch.from_numpy(class_mask).long()
        return img_tensor, mask_tensor

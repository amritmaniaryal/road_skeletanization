import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    """Load (road mask, centerline skeleton) PNG pairs produced by make_data.

    Pairs are stored as ``image_<id>.png`` / ``target_<id>.png`` inside
    ``data_dir``. A deterministic 80/20 split is derived from a seeded RNG so
    every run sees the same train / validation partition.

    Args:
        data_dir: directory containing the generated pairs.
        split: 'train' or 'val'.
        seed: seed for the train/val split.
        val_ratio: fraction of samples held out for validation.
        augment: apply random horizontal/vertical flips (train only).
    """

    def __init__(self, data_dir, split="train", seed=42, val_ratio=0.2, augment=False):
        ids = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("image_") and fname.endswith(".png"):
                ids.append(fname.replace("image_", "").replace(".png", ""))
        if not ids:
            raise FileNotFoundError(
                f"No image_*.png pairs found in {data_dir}. Run `python -m src.make_data` first."
            )

        rng = random.Random(seed)
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_ratio))
        if split == "train":
            self.ids = ids[n_val:]
        elif split == "val":
            self.ids = ids[:n_val]
        else:
            raise ValueError(f"Unknown split: {split!r}")

        self.data_dir = data_dir
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img = np.asarray(
            Image.open(os.path.join(self.data_dir, f"image_{sample_id}.png"))
        ).astype(np.float32) / 255.0
        target = np.asarray(
            Image.open(os.path.join(self.data_dir, f"target_{sample_id}.png"))
        ).astype(np.float32) / 255.0

        # (H, W) -> (1, H, W) in [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
                target = torch.flip(target, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
                target = torch.flip(target, dims=[1])

        return img, target

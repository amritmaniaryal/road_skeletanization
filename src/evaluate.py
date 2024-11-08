"""Evaluate a trained model on the validation set and save visualizations.

Usage:
    python -m src.evaluate --config configs/smoke.yaml \
        --ckpt checkpoints/best.pth --out runs/eval_smoke
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import RoadDataset
from src.metrics import dice_score, iou, mse, skeleton_metrics
from src.model import UNet
from src.utils import pick_device

from PIL import Image


def save_overlay(imgs, gts, preds, out_dir, prefix="sample"):
    """Side-by-side: input mask | ground-truth skeleton | predicted skeleton."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(imgs), 8)
    for i in range(n):
        img = imgs[i, 0].cpu().numpy()
        gt = gts[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()

        canvas = np.zeros((img.shape[0], img.shape[1] * 3, 3), dtype=np.uint8)
        # input mask (grayscale)
        canvas[:, :img.shape[1], :] = (img * 255).astype(np.uint8)[..., None]
        # ground truth skeleton (green)
        canvas[:, img.shape[1]:2 * img.shape[1], 1] = (gt * 255).astype(np.uint8)
        # prediction (red)
        canvas[:, 2 * img.shape[1]:, 0] = (pred * 255).astype(np.uint8)

        fp = out_dir / f"{prefix}_{i:03d}.png"
        Image.fromarray(canvas).save(fp)
    print(f"Saved {n} overlay images to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best.pth")
    parser.add_argument("--out", default="runs/eval_smoke")
    parser.add_argument("--tol", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = pick_device(cfg.get("device"))
    print(f"Using device: {device}")

    val_ds = RoadDataset(
        cfg.get("data_dir", "data/thinning"), split="val", seed=cfg["seed"],
        val_ratio=cfg.get("val_ratio", 0.2),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False)

    model = UNet(base_ch=cfg.get("base_ch", 32), depth=cfg.get("depth", 3))
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    total = {"dice": 0.0, "iou": 0.0, "mse": 0.0,
             "prec": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0
    all_preds, all_gts, all_imgs = [], [], []

    with torch.no_grad():
        for imgs, gts in val_loader:
            imgs, gts = imgs.to(device), gts.to(device)
            preds = model(imgs)
            total["dice"] += dice_score(preds, gts).item()
            total["iou"] += iou(preds, gts).item()
            total["mse"] += mse(preds, gts).item()
            p, r, f = skeleton_metrics(preds, gts, tol=args.tol)
            total["prec"] += p
            total["recall"] += r
            total["f1"] += f

            all_imgs.append(imgs.cpu())
            all_gts.append(gts.cpu())
            all_preds.append(preds.cpu())
            n_batches += 1

    n = max(n_batches, 1)
    print(f"Evaluated {len(val_ds)} validation samples:")
    print(f"  Dice       : {total['dice'] / n:.4f}")
    print(f"  IoU        : {total['iou'] / n:.4f}")
    print(f"  MSE        : {total['mse'] / n:.4f}")
    print(f"  Prec(tol={args.tol}) : {total['prec'] / n:.4f}")
    print(f"  Recall(tol={args.tol}): {total['recall'] / n:.4f}")
    print(f"  F1(tol={args.tol})   : {total['f1'] / n:.4f}")

    save_overlay(
        torch.cat(all_imgs), torch.cat(all_gts), torch.cat(all_preds),
        Path(args.out),
    )


if __name__ == "__main__":
    main()

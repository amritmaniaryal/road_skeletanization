"""Benchmark: learned U-Net vs classical skeletonizers across noise levels.

For every noise intensity and every method, we run on the same corrupted
validation inputs, score against the clean OSM skeleton ground truth, and
write a results table plus side-by-side overlays.

Usage:
    python -m src.benchmark --config configs/benchmark.yaml
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.baselines import BASELINES
from src.corrupt import corrupt
from src.data.dataset import RoadDataset
from src.metrics import dice_score, iou, mse, skeleton_metrics
from src.model import UNet
from src.utils import pick_device


def run_method(name, mask_np, model, device):
    """Produce a binary skeleton prediction from a binary (H, W) mask."""
    if name == "unet":
        x = torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device)
        with torch.no_grad():
            prob = model(x)[0, 0].cpu().numpy()
        return (prob > 0.5).astype(np.uint8)
    return BASELINES[name](mask_np)


def save_overlay(mask, gt, preds_by_method, out_dir, idx):
    """Columns: corrupted mask | GT skeleton | each method's output."""
    methods = list(preds_by_method.keys())
    n_cols = 1 + 1 + len(methods)
    canvas = np.zeros((mask.shape[0], mask.shape[1] * n_cols, 3), dtype=np.uint8)

    canvas[:, :mask.shape[1], :] = (mask * 255).astype(np.uint8)[..., None]
    x0 = mask.shape[1]
    canvas[:, x0:2 * x0, 1] = (gt * 255).astype(np.uint8)
    for i, mname in enumerate(methods):
        c = 2 + i
        canvas[:, c * x0:(c + 1) * x0, 0] = (preds_by_method[mname] * 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_dir / f"overlay_{idx:03d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = pick_device(cfg.get("device"))
    print(f"Using device: {device}")

    data_dir = cfg.get("data_dir", "data/thinning")
    noise_levels = cfg.get("noise_levels", [0.0, 0.3, 0.6, 0.9])
    methods = cfg.get("methods", ["unet", "medial_axis", "skeletonize", "distance_ridge"])
    tol = cfg.get("tol", 2)
    noise_seed = cfg.get("noise_seed", 42)

    val_ds = RoadDataset(
        data_dir, split="val", seed=cfg["seed"],
        val_ratio=cfg.get("val_ratio", 0.2),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet(base_ch=cfg.get("base_ch", 32), depth=cfg.get("depth", 3))
    state = torch.load(cfg.get("ckpt", "checkpoints/best.pth"), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    # metric helpers aggregated per sample
    def agg(metrics_dict):
        n = len(metrics_dict)
        return {k: sum(v[k] for v in metrics_dict) / n for k in metrics_dict[0]}

    rows = []
    overlays_dir = Path(cfg.get("out_dir", "runs/benchmark"))
    for intensity in noise_levels:
        print(f"\n=== noise intensity {intensity} ===")
        per_method = {m: [] for m in methods}
        is_heaviest = intensity == noise_levels[-1]

        for i, (_, gt_t) in enumerate(val_loader):
            gt = gt_t[0, 0].numpy()
            mask_clean = np.asarray(
                Image.open(val_ds.data_dir + f"/image_{val_ds.ids[i]}.png")
            ).astype(np.float32) / 255.0
            mask = corrupt(mask_clean, {"intensity": intensity}, seed=noise_seed + i)

            gt_bin = (gt > 0.5).astype(np.float32)
            for mname in methods:
                sk = run_method(mname, mask, model, device)
                sk_t = torch.from_numpy(sk.astype(np.float32))[None, None]
                dice = dice_score(sk_t, torch.from_numpy(gt_bin)[None, None]).item()
                iou_v = iou(sk_t, torch.from_numpy(gt_bin)[None, None]).item()
                mse_v = mse(sk_t, torch.from_numpy(gt_bin)[None, None]).item()
                p, r, f = skeleton_metrics(sk_t, torch.from_numpy(gt_bin)[None, None], tol=tol)
                per_method[mname].append(
                    {"dice": dice, "iou": iou_v, "mse": mse_v, "prec": p, "recall": r, "f1": f}
                )

            if is_heaviest and i < 8:
                preds = {mname: run_method(mname, mask, model, device) for mname in methods}
                save_overlay(mask, (gt > 0.5).astype(np.uint8), preds, overlays_dir, i)

        for mname in methods:
            a = agg(per_method[mname])
            rows.append(
                {
                    "noise": intensity, "method": mname,
                    "dice": round(a["dice"], 4), "iou": round(a["iou"], 4),
                    "mse": round(a["mse"], 5), "prec": round(a["prec"], 4),
                    "recall": round(a["recall"], 4), "f1": round(a["f1"], 4),
                }
            )
            print(f"  {mname:15s} dice={a['dice']:.3f} iou={a['iou']:.3f} "
                  f"f1(tol={tol})={a['f1']:.3f}")

    out_dir = Path(cfg.get("out_dir", "runs/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote results to {csv_path}")
    print(f"Overlays in {out_dir}/")


if __name__ == "__main__":
    main()

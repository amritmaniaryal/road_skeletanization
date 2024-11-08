import argparse
from pathlib import Path

import torch
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import RoadDataset
from src.metrics import BCEWithDice, dice_score, iou
from src.model import UNet
from src.utils import pick_device, set_seed


def main(cfg):
    set_seed(cfg["seed"])
    device = pick_device(cfg.get("device"))
    print(f"Using device: {device}")

    data_dir = cfg.get("data_dir", "data/thinning")
    batch_size = cfg.get("batch_size", 4)

    train_ds = RoadDataset(
        data_dir, split="train", seed=cfg["seed"],
        val_ratio=cfg.get("val_ratio", 0.2), augment=cfg.get("augment", True),
    )
    val_ds = RoadDataset(
        data_dir, split="val", seed=cfg["seed"], val_ratio=cfg.get("val_ratio", 0.2),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = UNet(
        base_ch=cfg.get("base_ch", 32), depth=cfg.get("depth", 3),
    ).to(device)
    loss_fn = BCEWithDice(dice_weight=cfg.get("dice_weight", 0.5))
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    best_val_loss = float("inf")
    checkpoints = Path(cfg.get("ckpt_dir", "checkpoints"))
    checkpoints.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.get("epochs", 1) + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, gts in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]"):
            imgs, gts = imgs.to(device), gts.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, gts in tqdm.tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [val]"):
                imgs, gts = imgs.to(device), gts.to(device)
                preds = model(imgs)
                val_loss += loss_fn(preds, gts).item()
                val_dice += dice_score(preds, gts).item()
                val_iou += iou(preds, gts).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        val_dice /= max(n_val, 1)
        val_iou /= max(n_val, 1)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = checkpoints / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "config": cfg,
                },
                ckpt,
            )
            print(f"  saved best checkpoint -> {ckpt}")

    # always save final weights too
    torch.save(model.state_dict(), checkpoints / "last.pth")
    print(f"Done. Final checkpoint at {checkpoints / 'last.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)

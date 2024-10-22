import argparse, torch, torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from .utils import set_seed                # relative import
from .data.dataset import RoadDataset
from .model import TinyUNet

import yaml, tqdm

def main(cfg):
    set_seed(cfg["seed"])

    # ---------- data ----------
    tr = T.Compose([T.ToTensor()])      # later: add noise, flips …
    train_ds = RoadDataset(split="train")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # ---------- model ----------
    model = TinyUNet().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.BCELoss()

    # ---------- loop ----------
    for epoch in range(cfg["epochs"]):
        for imgs, gts in tqdm.tqdm(train_loader):
            imgs, gts = imgs.cuda(), gts.cuda()
            preds = model(imgs)
            loss  = loss_fn(preds, gts)
            optim.zero_grad(); loss.backward(); optim.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # save weights
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baseline.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)

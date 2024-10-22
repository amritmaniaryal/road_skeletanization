import torch.nn as nn

class TinyUNet(nn.Module):
    """Placeholder – replace with real U-Net later."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

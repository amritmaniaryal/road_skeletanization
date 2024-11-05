import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Down(nn.Module):
    """MaxPool + DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    """Bilinear upsampling + skip-connection concat + DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net for binary segmentation (e.g. road mask -> centerline).

    Configurable channel base so it stays fast on CPU while still a real
    encoder/decoder with skip connections.
    """

    def __init__(self, in_ch=1, out_ch=1, base_ch=32, depth=3):
        super().__init__()
        self.inc = _DoubleConv(in_ch, base_ch)

        down_chs = [base_ch * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        in_ = base_ch
        for c in down_chs:
            self.downs.append(_Down(in_, c))
            in_ = c

        bridge_ch = down_chs[-1] * 2
        self.bridge = _DoubleConv(down_chs[-1], bridge_ch)

        # Decoder pairs with encoder skips [d_{depth-1}, ..., d_0, inc];
        # each up-block matches its skip resolution, ending back at input size.
        skip_chs = list(reversed(down_chs))[1:] + [base_ch]
        self.ups = nn.ModuleList()
        out_ = bridge_ch
        for skip_ch in skip_chs:
            self.ups.append(_Up(out_, skip_ch, skip_ch))
            out_ = skip_ch

        self.outc = nn.Sequential(
            nn.Conv2d(out_, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = self.bridge(skips[-1])

        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        return self.outc(x)

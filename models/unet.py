import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class RestoreNet(nn.Module):
    """
    Joint denoising (speckle + gaussian) + 2x super-resolution in a single pass.

    Input:  (B, 1, H, W)   degraded LR image (e.g. 128x128 or 256x256)
    Output: (B, 1, 2H, 2W) restored HR image (e.g. 256x256 or 512x512)

    Since the dataset always halves resolution (512->256 or 256->128),
    a fixed scale=2 upsample handles both cases.
    """

    def __init__(self, base_ch=64, n_resblocks=8, scale=2):
        super().__init__()
        self.head = nn.Conv2d(1, base_ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res
        up = self.upsample(feat)
        out = self.tail(up)
        return out


if __name__ == "__main__":
    # quick shape sanity check
    m = RestoreNet()
    dummy = torch.randn(2, 1, 128, 128)
    out = m(dummy)
    print("input:", dummy.shape, "-> output:", out.shape)  # expect (2,1,256,256)

"""UNet for the end-to-end deblur/denoise method. 4 downsampling levels, skip
connections carry the fine spatial detail across the bottleneck. Predicts a residual
added to the input observation (x_hat = y + UNet(y)) rather than the clean image
directly, since y is already close to x -- standard practice in denoising networks,
it gives the optimizer a much easier target than reconstructing from scratch."""
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 48):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c4, c4)

        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = conv_block(c4 + c4, c3)
        self.up3 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.dec3 = conv_block(c3 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.dec2 = conv_block(c2 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, 2, stride=2)
        self.dec1 = conv_block(c1 + c1, c1)

        self.out_conv = nn.Conv2d(c1, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(y)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return y + self.out_conv(d1)

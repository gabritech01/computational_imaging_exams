"""Learned Primal-Dual (PD-Net) with explicit TV structure: unrolls the
Chambolle-Pock algorithm for min_x 1/2||Ax-y||^2 + lambda*TV(x), replacing the exact
proximal operators with small learned CNN blocks. The dual variable lives in the
gradient space (via the Gradient operator), which is what ties this network to TV
regularization specifically, rather than a generic learned prior in data space."""
import torch
import torch.nn as nn

from src.degradation.operators import Blurring, Gradient


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class PDNet(nn.Module):
    def __init__(self, blur: Blurring, channels: int = 3, num_iterations: int = 8, mid_channels: int = 32):
        super().__init__()
        self.blur = blur
        self.grad = Gradient()
        self.num_iterations = num_iterations
        self.channels = channels

        dual_ch = 2 * channels
        self.dual_nets = nn.ModuleList([
            CNNBlock(dual_ch * 2, dual_ch, mid_channels) for _ in range(num_iterations)
        ])
        self.primal_nets = nn.ModuleList([
            CNNBlock(channels * 3, channels, mid_channels) for _ in range(num_iterations)
        ])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.blur._adjoint(y)
        x_bar = x
        p = torch.zeros(y.shape[0], 2 * self.channels, y.shape[2], y.shape[3], device=y.device, dtype=y.dtype)

        for k in range(self.num_iterations):
            gx_bar = self.grad._matvec(x_bar)
            p = p + self.dual_nets[k](torch.cat([p, gx_bar], dim=1))

            data_grad = self.blur._adjoint(self.blur._matvec(x) - y)
            gtp = self.grad._adjoint(p)
            x_new = x + self.primal_nets[k](torch.cat([x, data_grad, gtp], dim=1))

            x_bar = x_new + (x_new - x)
            x = x_new

        return x

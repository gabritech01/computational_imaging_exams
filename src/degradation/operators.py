"""Forward operators. the generic
differentiable Operator base class, the Gaussian blur operator and the discrete
gradient operator (needed for TV in the PD-Net method). CT-specific code (astra,
CTProjector, DownScaling) is dropped since it plays no role here.
"""
import math

import torch
import torch.nn.functional as F


class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x):
        ctx.op = op
        return op._matvec(x)

    @staticmethod
    def backward(ctx, grad_output):
        return None, ctx.op._adjoint(grad_output)


class Operator:
    """Base class for a linear operator A, exposing A(x) and its adjoint A.T(y)
    as autograd-aware calls so the operator can sit inside a training loop."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return OperatorFunction.apply(self, x)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return self._adjoint(y)

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Blurring(Operator):
    """Gaussian blur, applied as a depthwise convolution with circular (periodic)
    boundary conditions. This makes A a circulant matrix, which has two consequences
    that matter for the solvers built on top of it: the adjoint A^T is exactly equal
    to A itself (the gaussian kernel is symmetric), and the operator norm needed for
    FISTA's step size can be read off the kernel spectrum. Periodic boundaries are the
    standard simplifying assumption in the deblurring literature for this reason; the
    cost is a bit of wraparound ringing at the image border, which is minor here since
    the kernel is small (size 9) relative to a 256x256 image."""

    def __init__(self, channels: int, kernel_size: int, sigma: float):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.kernel = self._gaussian_kernel(kernel_size, sigma)
        self.pad = kernel_size // 2

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel

    def _weight(self, device, dtype):
        # depthwise kernel: same gaussian filter applied independently per channel
        return self.kernel.to(device=device, dtype=dtype).expand(self.channels, 1, -1, -1)

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        w = self._weight(x.device, x.dtype)
        x = F.pad(x, (self.pad,) * 4, mode="circular")
        return F.conv2d(x, w, groups=self.channels)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # symmetric kernel + circular boundary => the adjoint is the operator itself
        return self._matvec(y)


class Gradient(Operator):
    """Discrete image gradient (forward differences), used by the TV prior. Acts on
    (N, C, H, W) tensors and returns (N, 2*C, H, W): horizontal and vertical
    derivatives stacked on the channel axis."""

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(x)
        gx[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
        gy[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
        return torch.cat((gx, gy), dim=1)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        c = y.shape[1] // 2
        p, q = y[:, :c], y[:, c:]
        out = torch.zeros_like(p)
        out[..., :-1, :] -= p[..., :-1, :]
        out[..., 1:, :] += p[..., :-1, :]
        out[..., :, :-1] -= q[..., :, :-1]
        out[..., :, 1:] += q[..., :, :-1]
        return out

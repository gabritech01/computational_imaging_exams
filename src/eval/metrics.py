"""PSNR/SSIM on RGB images in [0,1]. Adapted from IPPy's utilities/metrics.py, which
only covered single-channel grayscale (1,1,H,W) tensors; here I work directly on
(C,H,W) and use skimage's multichannel SSIM."""
import math

import torch
from skimage.metrics import structural_similarity as skimage_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return -10 * math.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()
    return skimage_ssim(pred_np, target_np, data_range=1.0, channel_axis=2)

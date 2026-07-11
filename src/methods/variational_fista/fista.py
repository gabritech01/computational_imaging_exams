"""FISTA (Beck & Teboulle, 2009) for the deblurring problem regularized by wavelet
sparsity: min_x 1/2||Ax-y||^2 + lambda*||Wx||_1, with W an orthogonal wavelet
transform. Since W is orthogonal, the proximal operator of lambda*||Wx||_1 reduces to
soft-thresholding the wavelet coefficients directly (see NOTE_ORALE.md)."""
import numpy as np
import pywt
import torch

from src.degradation.operators import Blurring


def _threshold_channel(img: np.ndarray, lam: float, wavelet: str, level: int) -> np.ndarray:
    coeffs = pywt.wavedec2(img, wavelet, mode="periodization", level=level)
    approx, details = coeffs[0], coeffs[1:]
    shrunk = [approx] + [
        tuple(pywt.threshold(c, lam, mode="soft") for c in triplet) for triplet in details
    ]
    return pywt.waverec2(shrunk, wavelet, mode="periodization")


def wavelet_soft_threshold(x: torch.Tensor, lam: float, wavelet: str = "db4", level: int = 3) -> torch.Tensor:
    """Applies a per-channel 2D DWT, soft-thresholds the detail coefficients only
    (the coarse approximation is left untouched), and reconstructs. x: (C,H,W)."""
    arr = x.detach().cpu().numpy()
    out = np.stack([_threshold_channel(arr[c], lam, wavelet, level) for c in range(arr.shape[0])])
    return torch.from_numpy(out).to(x.dtype)


def fista_deblur(y: torch.Tensor, blur: Blurring, lam: float, n_iter: int = 100,
                  wavelet: str = "db4", level: int = 3) -> torch.Tensor:
    """Solves the problem for a single (C,H,W) observation y. The step size is 1/L
    with L=1: under circular boundary conditions the gaussian blur is a symmetric
    circulant operator, whose largest singular value is exactly 1 (the kernel sums to
    1, with the maximum attained at the zero frequency), so no power iteration is
    needed to estimate L."""
    x_prev = y.clone()
    z = y.clone()
    t_prev = 1.0
    for _ in range(n_iter):
        grad = blur._adjoint(blur._matvec(z.unsqueeze(0)) - y.unsqueeze(0))[0]
        x = wavelet_soft_threshold(z - grad, lam, wavelet, level)
        t = (1 + (1 + 4 * t_prev**2) ** 0.5) / 2
        z = x + ((t_prev - 1) / t) * (x - x_prev)
        x_prev, t_prev = x, t
    return x_prev.clamp(0, 1)

"""Picks the best available torch device: CUDA (Colab), then MPS (Apple Silicon), CPU last."""
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

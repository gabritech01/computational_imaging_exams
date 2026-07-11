"""Builds the degraded observations that every method will be evaluated on. Run once
per configuration (full-resolution and the low-res variant for the generative method);
the resulting PNGs on disk are the fixed inputs every method reads from afterwards,
which is what enforces the "same degraded input for all methods" requirement.

Saving to 8-bit PNG (instead of keeping raw float tensors) is a deliberate choice: it
quantizes y the way a real acquisition would, and guarantees every method reads the
exact same bytes regardless of what machine or torch version generated them.
"""
import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image

from src.data.kaokore import KaokoreDataset
from src.degradation.operators import Blurring

NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]
BASE_SEED = 0


def degrade(x: torch.Tensor, blur: Blurring, noise_level: float, seed: int) -> torch.Tensor:
    y = blur(x)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(y.shape, generator=generator) * noise_level
    return (y + noise).clamp(0, 1)


def run(root: str, out_dir: str, image_size: int, kernel_size: int, sigma: float,
        n_dev: int | None, n_test: int | None):
    blur = Blurring(channels=3, kernel_size=kernel_size, sigma=sigma)
    subset_size = {"dev": n_dev, "test": n_test}

    for split in ["dev", "test"]:
        dataset = KaokoreDataset(root, split=split, image_size=image_size)

        n = subset_size[split]
        if n is not None:
            # fixed, deterministic subset: same 80/20 images every time this is run,
            # picked with a dedicated generator so it never depends on global RNG state
            g = torch.Generator().manual_seed(BASE_SEED)
            keep = torch.randperm(len(dataset), generator=g)[:n].tolist()
            dataset.entries = [dataset.entries[i] for i in keep]

        clean_dir = os.path.join(out_dir, split, "clean")
        os.makedirs(clean_dir, exist_ok=True)
        for level in NOISE_LEVELS:
            os.makedirs(os.path.join(out_dir, split, f"sigma_{level}"), exist_ok=True)

        for idx in range(len(dataset)):
            x, _ = dataset[idx]
            filename = dataset.entries[idx]["image"]
            to_pil_image(x).save(os.path.join(clean_dir, filename))

            for level_idx, level in enumerate(NOISE_LEVELS):
                seed = BASE_SEED + idx * len(NOISE_LEVELS) + level_idx
                y = degrade(x.unsqueeze(0), blur, level, seed)[0]
                to_pil_image(y).save(os.path.join(out_dir, split, f"sigma_{level}", filename))

        print(f"{split}: {len(dataset)} images degraded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/kaokore")
    parser.add_argument("--out", default="data/degraded")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--n_dev", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    args = parser.parse_args()
    run(args.root, args.out, args.image_size, args.kernel_size, args.sigma, args.n_dev, args.n_test)

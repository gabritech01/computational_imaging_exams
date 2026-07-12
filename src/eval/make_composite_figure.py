"""Builds a single figure comparing all 3 methods side by side on the same test image,
for two noise levels, for the presentation's results slide."""
import os

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

IMAGE_NAME = "00000700.jpg"
LEVELS = [0.01, 0.1]


def load(path):
    return to_tensor(Image.open(path).convert("RGB")).permute(1, 2, 0).numpy()


def run():
    fig, axes = plt.subplots(len(LEVELS), 5, figsize=(15, 3.2 * len(LEVELS)))
    cols = ["originale", "degradata", "FISTA-Wavelet", "PD-Net", "UNet"]

    for row, level in enumerate(LEVELS):
        clean = load(f"data/degraded/test/clean/{IMAGE_NAME}")
        degraded = load(f"data/degraded/test/sigma_{level}/{IMAGE_NAME}")
        fista = load(f"results/fista/sigma_{level}/reconstructed/{IMAGE_NAME}")
        pdnet = load(f"results/pdnet/sigma_{level}/reconstructed/{IMAGE_NAME}")
        unet = load(f"results/unet/sigma_{level}/reconstructed/{IMAGE_NAME}")

        for col, (img, title) in enumerate(zip([clean, degraded, fista, pdnet, unet], cols)):
            ax = axes[row, col]
            ax.imshow(img.clip(0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=12)
        axes[row, 0].text(-0.15, 0.5, f"sigma={level}", transform=axes[row, 0].transAxes,
                           fontsize=12, rotation=90, va="center", ha="center")

    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/composite_comparison.png", dpi=150)
    print("saved results/figures/composite_comparison.png")


if __name__ == "__main__":
    run()

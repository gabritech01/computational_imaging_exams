"""Zoomed crop + |x - x_hat| difference maps at sigma=0.1, for the results slide.
Addresses the trace's explicit suggestion: "show some meaningful crops or, for
example, difference images between the output and the ground truth"."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

IMAGE_NAME = "00000700.jpg"
LEVEL = 0.1
CROP = (60, 160, 60, 160)  # top, bottom, left, right -- hair/edge region


def load(path):
    return to_tensor(Image.open(path).convert("RGB")).permute(1, 2, 0).numpy()


def crop(img, box):
    t, b, l, r = box
    return img[t:b, l:r]


def run():
    clean = load(f"data/degraded/test/clean/{IMAGE_NAME}")
    degraded = load(f"data/degraded/test/sigma_{LEVEL}/{IMAGE_NAME}")
    fista = load(f"results/fista/sigma_{LEVEL}/reconstructed/{IMAGE_NAME}")
    pdnet = load(f"results/pdnet/sigma_{LEVEL}/reconstructed/{IMAGE_NAME}")
    unet = load(f"results/unet/sigma_{LEVEL}/reconstructed/{IMAGE_NAME}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    cols = ["originale", "degradata", "FISTA-Wavelet", "PD-Net", "UNet"]
    imgs = [clean, degraded, fista, pdnet, unet]

    for col, (img, title) in enumerate(zip(imgs, cols)):
        axes[0, col].imshow(crop(img, CROP).clip(0, 1))
        axes[0, col].axis("off")
        axes[0, col].set_title(title, fontsize=11)

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    for col, (img, title) in enumerate(zip(imgs[2:], cols[2:]), start=2):
        diff = np.abs(img - clean).mean(axis=2)
        im = axes[1, col].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
        axes[1, col].axis("off")
        axes[1, col].set_title(f"|x - x̂| ({title})", fontsize=10)

    axes[0, 0].text(-0.15, 0.5, "crop\n(zoom)", transform=axes[0, 0].transAxes,
                     fontsize=10, rotation=90, va="center", ha="center")
    axes[1, 2].text(-0.4, 0.5, "mappa\nerrore", transform=axes[1, 2].transAxes,
                     fontsize=10, rotation=90, va="center", ha="center")

    fig.suptitle(f"sigma = {LEVEL}: crop ingrandito e mappe di errore assoluto", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/figures/crop_diff_comparison.png", dpi=150)
    print("saved results/figures/crop_diff_comparison.png")


if __name__ == "__main__":
    run()

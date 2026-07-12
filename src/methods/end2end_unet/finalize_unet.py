"""Evaluates the 4 trained UNet checkpoints (one per noise level) on the fixed test
subset. Same structure as finalize_fista.py: full reconstructions, a few side-by-side
comparison figures, per-level summary statistics, average inference time."""
import csv
import glob
import os
import statistics
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from src.eval.metrics import psnr, ssim
from src.methods.end2end_unet.model import UNet
from src.utils.device import get_device

NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]
RESULTS_DIR = "results"
N_EXAMPLES = 3


def load_test_pairs(level: float):
    clean_dir = "data/degraded/test/clean"
    deg_dir = f"data/degraded/test/sigma_{level}"
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.jpg")))
    pairs = []
    for name in names:
        x = to_tensor(Image.open(os.path.join(clean_dir, name)).convert("RGB"))
        y = to_tensor(Image.open(os.path.join(deg_dir, name)).convert("RGB"))
        pairs.append((name, x, y))
    return pairs


def run():
    device = get_device()
    recon_root = os.path.join(RESULTS_DIR, "unet")
    comp_dir = os.path.join(recon_root, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    summary_rows = []
    all_times = []

    for level in NOISE_LEVELS:
        model = UNet().to(device)
        ckpt = torch.load(f"results/checkpoints/unet_sigma_{level}.pt", map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        pairs = load_test_pairs(level)
        recon_dir = os.path.join(recon_root, f"sigma_{level}", "reconstructed")
        os.makedirs(recon_dir, exist_ok=True)

        psnrs, ssims = [], []
        with torch.no_grad():
            for i, (name, x, y) in enumerate(pairs):
                t0 = time.time()
                rec = model(y.unsqueeze(0).to(device)).clamp(0, 1).cpu()[0]
                elapsed = time.time() - t0
                all_times.append(elapsed)

                to_pil_image(rec).save(os.path.join(recon_dir, name))
                psnrs.append(psnr(rec, x))
                ssims.append(ssim(rec, x))

                if i < N_EXAMPLES:
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
                    for ax, img, title in zip(
                        axes, [x, y, rec], ["originale", f"degradata (sigma={level})", "ricostruita UNet"]
                    ):
                        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                        ax.set_title(title, fontsize=9)
                        ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(comp_dir, f"sigma_{level}_{name}.png"), dpi=150)
                    plt.close(fig)

        summary_rows.append({
            "noise_level": level,
            "psnr_mean": statistics.mean(psnrs),
            "psnr_std": statistics.stdev(psnrs),
            "ssim_mean": statistics.mean(ssims),
            "ssim_std": statistics.stdev(ssims),
        })
        print(f"level {level}: psnr {summary_rows[-1]['psnr_mean']:.3f}+-{summary_rows[-1]['psnr_std']:.3f}, "
              f"ssim {summary_rows[-1]['ssim_mean']:.4f}+-{summary_rows[-1]['ssim_std']:.4f}")

    avg_time = statistics.mean(all_times)
    print(f"avg reconstruction time per image: {avg_time:.4f}s over {len(all_times)} runs ({device})")

    with open(os.path.join(RESULTS_DIR, "tables", "unet_summary.csv"), "w", newline="") as f:
        fieldnames = ["noise_level", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        f.write(f"\navg_time_sec,{avg_time:.4f}\n")


if __name__ == "__main__":
    run()

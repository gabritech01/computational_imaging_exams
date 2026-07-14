"""Taro lambda per livello di rumore sul sottoinsieme dev, poi valuto FISTA sul
sottoinsieme test con i valori scelti. Scrivo results/tables/fista_results.csv e
qualche immagine ricostruita in results/figures/fista/ per il confronto visivo."""
import csv
import glob
import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from src.degradation.operators import Blurring
from src.eval.metrics import psnr, ssim
from src.methods.variational_fista.fista import fista_deblur

NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]
LAMBDA_GRID_FACTORS = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 4.0]
N_ITER = 100
DEGRADED_DIR = "data/degraded"
RESULTS_DIR = "results"


def load_pairs(split: str, level: float):
    clean_dir = os.path.join(DEGRADED_DIR, split, "clean")
    deg_dir = os.path.join(DEGRADED_DIR, split, f"sigma_{level}")
    filenames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.jpg")))
    pairs = []
    for name in filenames:
        x = to_tensor(Image.open(os.path.join(clean_dir, name)).convert("RGB"))
        y = to_tensor(Image.open(os.path.join(deg_dir, name)).convert("RGB"))
        pairs.append((name, x, y))
    return pairs


def tune_lambda(blur: Blurring, level: float) -> float:
    pairs = load_pairs("dev", level)
    best_lam, best_psnr = None, -1.0
    for factor in LAMBDA_GRID_FACTORS:
        lam = factor * level
        scores = [psnr(fista_deblur(y, blur, lam, N_ITER), x) for _, x, y in pairs]
        mean_psnr = sum(scores) / len(scores)
        if mean_psnr > best_psnr:
            best_psnr, best_lam = mean_psnr, lam
    return best_lam


def run():
    blur = Blurring(channels=3, kernel_size=9, sigma=2.0)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    fig_dir = os.path.join(RESULTS_DIR, "figures", "fista")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for level in NOISE_LEVELS:
        lam = tune_lambda(blur, level)
        pairs = load_pairs("test", level)
        for i, (name, x, y) in enumerate(pairs):
            rec = fista_deblur(y, blur, lam, N_ITER)
            rows.append({
                "noise_level": level, "lambda": lam, "image": name,
                "psnr": psnr(rec, x), "ssim": ssim(rec, x),
            })
            if i == 0:
                to_pil_image(rec).save(os.path.join(fig_dir, f"sigma_{level}_{name}"))
        print(f"level {level}: lambda={lam}")

    with open(os.path.join(RESULTS_DIR, "tables", "fista_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_level", "lambda", "image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()

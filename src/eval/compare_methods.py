"""Builds the final cross-method comparison: a combined table and the comparison plot
(PSNR/SSIM vs noise level, one curve per method) required by the project spec."""
import csv
import os

import matplotlib.pyplot as plt

NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]
METHODS = {
    "FISTA-Wavelet": "results/tables/fista_summary.csv",
    "PD-Net": "results/tables/pdnet_summary.csv",
    "UNet": "results/tables/unet_summary.csv",
}
TIMES = {"FISTA-Wavelet": None, "PD-Net": None, "UNet": None}


def read_summary(path: str):
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    rows = []
    for r in csv.DictReader(lines):
        try:
            r["noise_level"] = float(r["noise_level"])
        except (TypeError, ValueError):
            continue
        rows.append(r)
    time_line = [l for l in lines if l.startswith("avg_time_sec")]
    avg_time = float(time_line[0].split(",")[1]) if time_line else None
    return {r["noise_level"]: r for r in rows}, avg_time


def run():
    data = {}
    for method, path in METHODS.items():
        rows, avg_time = read_summary(path)
        data[method] = rows
        TIMES[method] = avg_time

    with open("results/tables/comparison_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "noise_level", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "avg_time_sec"])
        for method in METHODS:
            for level in NOISE_LEVELS:
                r = data[method][level]
                writer.writerow([method, level, r["psnr_mean"], r["psnr_std"],
                                  r["ssim_mean"], r["ssim_std"], TIMES[method]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        psnrs = [float(data[method][lv]["psnr_mean"]) for lv in NOISE_LEVELS]
        ssims = [float(data[method][lv]["ssim_mean"]) for lv in NOISE_LEVELS]
        axes[0].plot(NOISE_LEVELS, psnrs, marker="o", label=method)
        axes[1].plot(NOISE_LEVELS, ssims, marker="o", label=method)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("noise level")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs noise level")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("noise level")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs noise level")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/comparison_plot.png", dpi=150)
    print("saved results/figures/comparison_plot.png and results/tables/comparison_summary.csv")

    print("\navg inference time per image:")
    for method, t in TIMES.items():
        print(f"  {method}: {t:.4f}s")


if __name__ == "__main__":
    run()

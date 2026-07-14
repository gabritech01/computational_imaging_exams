"""Regenerates comparison_plot.png adding the degraded-observation baseline as a
dashed grey curve, so the plot shows what every method starts from and how much each
one gains. Baseline values computed from the saved degraded/clean PNGs."""
import csv

import matplotlib.pyplot as plt

LEVELS = [0.005, 0.01, 0.05, 0.1]
BASELINE_PSNR = [29.86, 29.81, 26.68, 21.57]
BASELINE_SSIM = [0.843, 0.840, 0.592, 0.297]


def read_methods():
    data = {}
    with open("results/tables/comparison_summary.csv") as f:
        for r in csv.DictReader(f):
            data.setdefault(r["method"], {"psnr": {}, "ssim": {}})
            data[r["method"]]["psnr"][float(r["noise_level"])] = float(r["psnr_mean"])
            data[r["method"]]["ssim"][float(r["noise_level"])] = float(r["ssim_mean"])
    return data


def run():
    methods = read_methods()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(LEVELS, BASELINE_PSNR, marker="s", linestyle="--", color="gray", label="Degradata (baseline)")
    axes[1].plot(LEVELS, BASELINE_SSIM, marker="s", linestyle="--", color="gray", label="Degradata (baseline)")
    for method in ["FISTA-Wavelet", "PD-Net", "UNet"]:
        axes[0].plot(LEVELS, [methods[method]["psnr"][lv] for lv in LEVELS], marker="o", label=method)
        axes[1].plot(LEVELS, [methods[method]["ssim"][lv] for lv in LEVELS], marker="o", label=method)

    axes[0].set(xscale="log", xlabel="noise level", ylabel="PSNR (dB)", title="PSNR vs noise level")
    axes[1].set(xscale="log", xlabel="noise level", ylabel="SSIM", title="SSIM vs noise level")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/figures/comparison_plot.png", dpi=150)
    print("saved results/figures/comparison_plot.png")


if __name__ == "__main__":
    run()

"""Artefatti finali per il metodo FISTA, da eseguire dopo che lambda è stato tarato
su dev e fista_results.csv contiene le metriche per-immagine sul test set per i
lambda scelti. Salvo tutte le immagini test ricostruite, qualche confronto visivo
affiancato, le statistiche riassuntive per livello (media+std) e il tempo medio di
ricostruzione."""
import csv
import os
import statistics
import time

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from src.degradation.operators import Blurring
from src.methods.variational_fista.fista import fista_deblur
from src.methods.variational_fista.run_fista import NOISE_LEVELS, N_ITER, load_pairs

RESULTS_DIR = "results"
N_EXAMPLES = 3


def lambdas_from_csv(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for level in NOISE_LEVELS:
        matching = [r for r in rows if float(r["noise_level"]) == level]
        out[level] = float(matching[0]["lambda"])
    return out


def run():
    lambdas = lambdas_from_csv(os.path.join(RESULTS_DIR, "tables", "fista_results.csv"))
    blur = Blurring(channels=3, kernel_size=9, sigma=2.0)

    recon_root = os.path.join(RESULTS_DIR, "fista")
    comp_dir = os.path.join(recon_root, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    summary_rows = []
    all_times = []

    for level in NOISE_LEVELS:
        lam = lambdas[level]
        pairs = load_pairs("test", level)
        recon_dir = os.path.join(recon_root, f"sigma_{level}", "reconstructed")
        os.makedirs(recon_dir, exist_ok=True)

        psnrs, ssims = [], []
        from src.eval.metrics import psnr, ssim

        for i, (name, x, y) in enumerate(pairs):
            t0 = time.time()
            rec = fista_deblur(y, blur, lam, N_ITER)
            elapsed = time.time() - t0
            all_times.append(elapsed)

            to_pil_image(rec).save(os.path.join(recon_dir, name))
            psnrs.append(psnr(rec, x))
            ssims.append(ssim(rec, x))

            if i < N_EXAMPLES:
                fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
                for ax, img, title in zip(
                    axes, [x, y, rec], ["originale", f"degradata (sigma={level})", "ricostruita FISTA"]
                ):
                    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                    ax.set_title(title, fontsize=9)
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(comp_dir, f"sigma_{level}_{name}.png"), dpi=150)
                plt.close(fig)

        summary_rows.append({
            "noise_level": level,
            "lambda": lam,
            "psnr_mean": statistics.mean(psnrs),
            "psnr_std": statistics.stdev(psnrs),
            "ssim_mean": statistics.mean(ssims),
            "ssim_std": statistics.stdev(ssims),
        })
        print(f"level {level}: psnr {summary_rows[-1]['psnr_mean']:.3f}+-{summary_rows[-1]['psnr_std']:.3f}, "
              f"ssim {summary_rows[-1]['ssim_mean']:.4f}+-{summary_rows[-1]['ssim_std']:.4f}")

    avg_time = statistics.mean(all_times)
    print(f"avg reconstruction time per image: {avg_time:.3f}s over {len(all_times)} runs")

    with open(os.path.join(RESULTS_DIR, "tables", "fista_summary.csv"), "w", newline="") as f:
        fieldnames = ["noise_level", "lambda", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        f.write(f"\navg_time_sec,{avg_time:.4f}\n")


if __name__ == "__main__":
    run()

"""Computes two things missing from the results, both from PNGs already on disk:
1. the degraded-observation baseline PSNR/SSIM per noise level (what every method
   starts from), and
2. per-image paired differences UNet vs PD-Net, so the 'indistinguishable' claim can
   be stated with the correct statistic (std of the paired differences, not the
   image-to-image std)."""
import glob
import os
import statistics

from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.eval.metrics import psnr, ssim

LEVELS = [0.005, 0.01, 0.05, 0.1]


def load(path):
    return to_tensor(Image.open(path).convert("RGB"))


def per_image(method_dir_tmpl, level):
    clean_dir = "data/degraded/test/clean"
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.jpg")))
    out = {}
    for name in names:
        x = load(os.path.join(clean_dir, name))
        rec = load(os.path.join(method_dir_tmpl.format(level=level), name))
        out[name] = psnr(rec, x)
    return out


def degraded_baseline():
    clean_dir = "data/degraded/test/clean"
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(clean_dir, "*.jpg")))
    print("=== baseline degradata (osservazione grezza vs pulita) ===")
    for level in LEVELS:
        deg_dir = f"data/degraded/test/sigma_{level}"
        ps, ss = [], []
        for name in names:
            x = load(os.path.join(clean_dir, name))
            y = load(os.path.join(deg_dir, name))
            ps.append(psnr(y, x))
            ss.append(ssim(y, x))
        print(f"sigma={level}: PSNR {statistics.mean(ps):.2f}+-{statistics.stdev(ps):.2f}  "
              f"SSIM {statistics.mean(ss):.3f}+-{statistics.stdev(ss):.3f}")


def paired_diffs():
    print("\n=== differenze appaiate UNet - PD-Net (per-immagine) ===")
    for level in LEVELS:
        unet = per_image("results/unet/sigma_{level}/reconstructed", level)
        pdnet = per_image("results/pdnet/sigma_{level}/reconstructed", level)
        diffs = [unet[n] - pdnet[n] for n in unet]
        mean_d = statistics.mean(diffs)
        sd_d = statistics.stdev(diffs)
        n = len(diffs)
        se = sd_d / (n ** 0.5)
        t = mean_d / se if se > 0 else float("inf")
        n_unet_better = sum(1 for d in diffs if d > 0)
        print(f"sigma={level}: diff media {mean_d:+.3f} dB, std diff {sd_d:.3f}, "
              f"SE {se:.3f}, t~{t:.2f}, UNet meglio in {n_unet_better}/{n} immagini")


if __name__ == "__main__":
    degraded_baseline()
    paired_diffs()

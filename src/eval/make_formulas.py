"""Renders the objective functions used in the slides as proper math typesetting
(matplotlib mathtext), instead of plain text, saved as transparent PNGs."""
import matplotlib.pyplot as plt

FORMULAS = {
    "formula_fista": r"$\min_x \ \frac{1}{2}\|Ax-y\|_2^2 \;+\; \lambda \|Wx\|_1$",
    "formula_pdnet": r"$\min_x \ \frac{1}{2}\|Ax-y\|_2^2 \;+\; \lambda \, \mathrm{TV}(x)$",
    "formula_unet_residual": r"$\hat{x} \;=\; y \;+\; \mathrm{UNet}(y)$",
}


def render(name: str, latex: str, fontsize: int = 30):
    fig = plt.figure(figsize=(8, 1.2))
    fig.text(0.02, 0.5, latex, fontsize=fontsize, va="center", ha="left", color="#202020")
    fig.savefig(f"results/figures/{name}.png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    for name, latex in FORMULAS.items():
        render(name, latex)
    print("saved formula images to results/figures/")

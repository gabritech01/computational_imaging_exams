"""Simple schematic diagrams of the UNet and PD-Net architectures for the slides."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch


def box(ax, x, y, w, h, text, color="#4472C4", fontsize=9, fontcolor="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor="black", linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=fontcolor)


def arrow(ax, x0, y0, x1, y1, color="black"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3))


def unet_diagram():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    enc_x = [0.5, 2.0, 3.5, 5.0]
    sizes = ["256x256x48", "128x128x96", "64x64x192", "32x32x384"]
    for i, (x, s) in enumerate(zip(enc_x, sizes)):
        y = 3.6 - i * 0.5
        box(ax, x, y, 1.3, 0.6, f"Enc {i+1}\n{s}", color="#4472C4")
        if i > 0:
            arrow(ax, enc_x[i - 1] + 0.65, (3.6 - (i - 1) * 0.5), x + 0.65, y + 0.6)

    box(ax, 6.3, 1.4, 1.4, 0.6, "Bottleneck\n16x16x384", color="#8E44AD")
    arrow(ax, 5.65, 3.6 - 3 * 0.5 + 0.6, 6.3, 1.4 + 0.3)

    dec_x = [7.9, 7.9, 7.9, 7.9]
    for i, s in enumerate(reversed(sizes)):
        y = 0.7 + i * 1.0
        box(ax, dec_x[i], y, 1.3, 0.6, f"Dec {4-i}\n{s}", color="#2E7D32")
        if i > 0:
            arrow(ax, dec_x[i] + 0.65, y - 0.4 + 0.6, dec_x[i] + 0.65, y)
    arrow(ax, 7.7, 1.4 + 0.3, 7.9, 0.7 + 0.3)

    for i, x in enumerate(enc_x):
        y_enc = 3.6 - i * 0.5 + 0.3
        y_dec = 0.7 + (3 - i) * 1.0 + 0.3
        ax.annotate("", xy=(7.9, y_dec), xytext=(x + 1.3, y_enc),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, linestyle="dashed",
                                     connectionstyle="arc3,rad=0.15"))

    box(ax, 9.5, 0.7, 1.0, 0.6, "out_conv\n1x1", color="#C0392B")
    arrow(ax, 9.2, 0.7 + 0.3, 9.5, 1.0)
    ax.text(10.5, 1.0, "x_hat =\ny + out", fontsize=9, ha="center")

    ax.text(0.2, 4.5, "input y (3x256x256)", fontsize=10)
    ax.text(2.5, 0.1, "skip connections (dashed)", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig("results/figures/unet_diagram.png", dpi=150)
    plt.close(fig)


def pdnet_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    box(ax, 0.3, 2.0, 1.3, 0.8, "x_k\n(primal)", color="#2E7D32")
    box(ax, 0.3, 0.6, 1.3, 0.8, "p_k\n(dual)", color="#4472C4")

    box(ax, 2.3, 0.6, 1.6, 0.8, "Gradient(x_bar)", color="#8E44AD", fontsize=8)
    box(ax, 4.4, 0.6, 1.8, 0.8, "DualNet_k\n[p_k, Gx_bar]", color="#4472C4", fontsize=8)
    arrow(ax, 1.6, 1.0, 2.3, 1.0)
    arrow(ax, 3.9, 1.0, 4.4, 1.0)
    box(ax, 6.7, 0.6, 1.3, 0.8, "p_{k+1}", color="#4472C4")
    arrow(ax, 6.2, 1.0, 6.7, 1.0)

    box(ax, 2.3, 2.4, 2.0, 0.8, "A^T(Ax_k - y)\n+ Gradient.T(p)", color="#8E44AD", fontsize=8)
    box(ax, 4.7, 2.4, 1.9, 0.8, "PrimalNet_k\n[x_k, grads]", color="#2E7D32", fontsize=8)
    arrow(ax, 1.6, 2.4, 2.3, 2.8)
    arrow(ax, 4.3, 2.8, 4.7, 2.8)
    box(ax, 7.0, 2.4, 1.3, 0.8, "x_{k+1}", color="#2E7D32")
    arrow(ax, 6.6, 2.8, 7.0, 2.8)
    arrow(ax, 7.35, 2.4, 7.35, 1.4)
    ax.text(7.5, 1.8, "extrapolation\nx_bar = x_new+(x_new-x)", fontsize=7, color="gray")

    ax.text(8.9, 2.5, "repeat for\nk = 1..8", fontsize=10, ha="center")
    ax.annotate("", xy=(0.3, 2.9), xytext=(8.3, 3.6),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1, connectionstyle="arc3,rad=-0.3"))

    ax.text(0.1, 4.1, "initialization: x_0 = A^T(y), p_0 = 0", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/pdnet_diagram.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    unet_diagram()
    pdnet_diagram()
    print("saved unet_diagram.png and pdnet_diagram.png")

"""OriginPro-style training curves for the manuscript (one 4-panel figure).

    python plot_training_curves.py   ->  ISIC_Dermavision_Manuscript/training_curves.{pdf,png}
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

OUT = Path("ISIC_Dermavision_Manuscript/training_curves")
plt.rcParams.update({
    "font.family": "Arial", "font.weight": "bold", "font.size": 11,
    "axes.labelweight": "bold", "axes.linewidth": 1.6, "axes.labelsize": 12,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.fontsize": 7.5,
})


def load(name):
    rows = list(csv.DictReader(open(f"results/{name}_epoch_history.csv")))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


# (marker, face colour) for training / validation, matching the authors' original Origin figures
STYLE = {
    ("MobileNetV2", "Accuracy"): (("p", "#CC33CC"), ("D", "#8B0000")),
    ("MobileNetV2", "Loss"): (("s", "#E04000"), ("o", "#33CCCC")),
    ("DINOv2", "Accuracy"): (("s", "#000080"), ("o", "#CC0000")),
    ("DINOv2", "Loss"): (("s", "#FF0000"), ("o", "#404040")),
}


def panel(ax, x, tr, va, tag, kind, model, tag_xy, legend_loc):
    (mt, ct), (mv, cv) = STYLE[(model, kind)]
    for y, mk, c, split in ((tr, mt, ct, "Training"), (va, mv, cv, "Validation")):
        ax.plot(x, y, "-", marker=mk, color=c, mfc=c, mec="black", mew=0.5, ms=4.5, lw=1.4,
                label=f"{model} {split} {kind}")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlim(0, 52)
    ax.set_xlabel("Epochs"); ax.set_ylabel(kind + (" (%)" if kind == "Accuracy" else ""))
    ax.grid(True, ls="--", lw=0.6, color="0.6")
    ax.text(*tag_xy, tag, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="center")
    ax.legend(loc=legend_loc)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


m, d = load("mobilenetv2"), load("dinov2")
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
panel(ax[0], m["Epoch"], m["Train_Accuracy"], m["Val_Accuracy"], "(a)", "Accuracy", "MobileNetV2", (0.9, 0.62), "lower right")
panel(ax[1], m["Epoch"], m["Train_Loss"], m["Val_Loss"], "(b)", "Loss", "MobileNetV2", (0.9, 0.62), "upper right")
panel(ax[2], d["Epoch"], d["Train_Accuracy"], d["Val_Accuracy"], "(c)", "Accuracy", "DINOv2", (0.1, 0.95), "center right")
panel(ax[3], d["Epoch"], d["Train_Loss"], d["Val_Loss"], "(d)", "Loss", "DINOv2", (0.9, 0.62), "upper right")
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))

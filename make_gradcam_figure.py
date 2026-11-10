"""Compact Grad-CAM figure for the manuscript: 4 cases x (original, overlay).

    python make_gradcam_figure.py  ->  ISIC_Dermavision_Manuscript/gradcam_figure.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC = Path("results/gradcam/gradcam_samples")
OUT = Path("ISIC_Dermavision_Manuscript/gradcam_figure")
CASES = [
    ("01_bcc_tp_ISIC_0073082", "(a) BCC\ncorrect"),
    ("02_melanoma_tp_ISIC_0033522", "(b) Melanoma\ncorrect"),
    ("03_nevus_tp_ISIC_0026951", "(c) Nevus\ncorrect"),
    ("04_melanoma_fn_as_nevus_ISIC_0025313", "(d) Melanoma\npredicted as Nevus"),
]
plt.rcParams.update({"font.family": "Arial", "font.weight": "bold", "font.size": 11})


def square(path):
    im = Image.open(path).convert("RGB")
    s = min(im.size); l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((400, 400), Image.LANCZOS)


fig, ax = plt.subplots(2, 4, figsize=(9, 5))
for c, (stem, title) in enumerate(CASES):
    for r, kind in enumerate(("original", "overlay")):
        a = ax[r, c]
        a.imshow(square(SRC / f"{stem}_{kind}.png"))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(1.6)
        if r == 0:
            a.set_title(title, fontweight="bold", color="#B00000" if "predicted" in title else "black")
ax[0, 0].set_ylabel("Input", fontweight="bold", fontsize=12)
ax[1, 0].set_ylabel("Grad-CAM", fontweight="bold", fontsize=12)
fig.tight_layout(h_pad=0.4, w_pad=0.4)
fig.savefig(OUT.with_suffix(".pdf"), dpi=300)
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"))

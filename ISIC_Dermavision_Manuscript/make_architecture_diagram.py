"""Generate dermavision_architecture.drawio: compact lane layout, native editable shapes.

Run from the repo root:
    python ISIC_Dermavision_Manuscript/make_architecture_diagram.py ISIC_Dermavision_Manuscript/dermavision_architecture.drawio
Then export PDF/PNG with the draw.io desktop CLI (-x -f pdf --crop / -x -f png -s 3)."""
import base64, io, sys
from xml.sax.saxutils import quoteattr
from PIL import Image

OUT = sys.argv[1]
IMG = "results/gradcam/gradcam_samples/"
RPI = "ISIC_Dermavision_Manuscript/raspberry_pi.png"
BCC, MEL, NEV = (IMG + f for f in ("01_bcc_tp_ISIC_0073082_original.png",
                                   "02_melanoma_tp_ISIC_0033522_original.png",
                                   "03_nevus_tp_ISIC_0026951_original.png"))
OVL = IMG + "02_melanoma_tp_ISIC_0033522_overlay.png"

cells, n = [], [1]
FONT = "fontFamily=Helvetica;fontColor=#000000;fontStyle=1;"
K = "#000000"
WHITE, INNER = ("#FFFFFF", K), ("#FFFFFF", "#000000")
MOB, DINO = ("#A8DDB1", K), ("#E8DDF7", K)
REC, BAD = ("#D5F0D8", "#2E7D32"), ("#FBE3E3", "#B23B3B")


def nid():
    n[0] += 1
    return f"n{n[0]}"


def box(x, y, w, h, text, c, parent="1", fs=11, extra=""):
    cid = nid()
    style = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor={c[0]};strokeColor={c[1]};"
             f"strokeWidth=1.5;fontSize={fs};{FONT}{extra}")
    cells.append(f'<mxCell id="{cid}" value={quoteattr(text)} style="{style}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def text(x, y, w, h, t, parent="1", fs=11, extra=""):
    return box(x, y, w, h, t, ("none", "none"), parent, fs, "rounded=0;" + extra)


def group(x, y, w, h, title, c, fs=14, align="center"):
    return box(x, y, w, h, f"<b>{title}</b>", c, fs=fs,
               extra=f"container=1;collapsible=0;verticalAlign=top;align={align};spacingLeft=10;"
                     "spacingTop=4;strokeWidth=3;arcSize=4;")


def img(x, y, w, h, path, parent="1", square=True, border=K):
    im = Image.open(path).convert("RGB")
    if square:
        s = min(im.size); l, t = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((200, 200), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cid = nid()
    cells.append(f'<mxCell id="{cid}" value="" style="shape=image;html=1;imageAspect=0;aspect=fixed;'
                 f'imageBorder={border};image=data:image/jpeg,{b64};" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def edge(s, t, parent="1", extra=""):
    cells.append(f'<mxCell id="{nid()}" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
                 f'endArrow=block;endFill=1;endSize=5;strokeColor=#000000;strokeWidth=2;{extra}" edge="1" '
                 f'parent="{parent}" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')


def card(parent, x, y, w, h, title, rows, fs=11):
    return box(x, y, w, h, f"<b>{title}</b><hr size='1'>" + "<br>".join(rows), WHITE, parent, fs,
               "align=left;spacingLeft=8;verticalAlign=top;spacingTop=4;")


TOP = 50

# ---- Left column: dataset -> preprocessing -> split ----
ds = group(20, TOP, 250, 160, "ISIC 2019 Dataset", WHITE)
for i, (p, cap) in enumerate([(BCC, "BCC<br>3,323"), (MEL, "Melanoma<br>4,522"), (NEV, "Nevus<br>4,600")]):
    img(17 + i * 76, 30, 64, 64, p, ds)
    text(9 + i * 76, 96, 80, 30, cap, ds, 10)
box(10, 128, 230, 24, "12,445 images, 3 classes", INNER, ds, 11)

pre = group(20, TOP + 188, 250, 96, "Preprocessing", WHITE)
chips = [box(8 + i * 82, 32, 62, 52, t, INNER, pre, 10) for i, t in enumerate(
    ["Resize<br>224 × 224", "Augment<br>flip, rotate, zoom", "Normalize<br>0–1"])]
edge(chips[0], chips[1], pre); edge(chips[1], chips[2], pre)

sp = group(20, TOP + 312, 250, 136, "Data Split", WHITE)
pie = "shape=mxgraph.basic.pie;strokeWidth=2;rounded=0;"
box(10, 32, 84, 84, "", ("#2F5FC4", "#FFFFFF"), sp, extra=pie + "startAngle=0.2;endAngle=1;")
box(10, 32, 84, 84, "", ("#2E7D32", "#FFFFFF"), sp, extra=pie + "startAngle=0;endAngle=0.2;")
text(18, 74, 40, 20, "80%", sp, 12, "fontColor=#FFFFFF;")
text(56, 40, 30, 18, "20%", sp, 9, "fontColor=#FFFFFF;")
box(104, 32, 136, 38, "Training 80%<br>9,956 images", ("#E3F0FB", "#2F5FC4"), sp, 10)
box(104, 76, 136, 38, "Validation 20%<br>2,489 images", ("#E3F4E1", "#2E7D32"), sp, 10)
text(100, 114, 145, 18, "Stratified, image-level", sp, 10, "fontStyle=3;fontColor=#000000;")
edge(ds, pre); edge(pre, sp)

# ---- Model lanes: architecture -> FP32 baseline -> optimization ----
LX, LW = 330, 1058
STEP_CY = 86  # vertical centre of step row inside a lane: arrows anchor here to run straight


def lane(y, h, title, c, steps):
    g = group(LX, y, LW, h, title, c, 14, "left")
    ids = [box(10 + i * 94, 50, 74, 72, t, INNER, g, 11) for i, t in enumerate(steps)]
    for a, b in zip(ids, ids[1:]):
        edge(a, b, g)
    text(10, 28, 540, 20, "Architecture", g, 11, "align=left;fontColor=#000000;")
    text(584, 28, 180, 20, "Baseline FP32 evaluation", g, 11, "align=left;fontColor=#000000;")
    text(794, 28, 250, 20, "Optimization experiments", g, 11, "align=left;fontColor=#000000;")
    return g, ids


mob, m = lane(TOP, 232, "Branch 1: MobileNetV2 (CNN, ~2.6M parameters)", MOB, [
    "Input<br>224 × 224", "MobileNetV2<br>(ImageNet)", "Global Avg<br>Pooling",
    "Dense 256<br>ReLU", "Dense 64<br>ReLU", "Dense 3<br>Softmax"])
ev = card(mob, 584, 50, 180, 104, "MobileNetV2 FP32", [
    "Accuracy: 90.96%", "Macro F1: 91.27%", "Size: 30.21 MiB", "CPU latency: 104.89 ms"])
op = box(794, 50, 254, 172, "", INNER, mob)
box(8, 8, 238, 30, "Dynamic INT8 TFLite: 90.88%", WHITE, op, 11)
box(8, 44, 238, 44, "Full INT8 TFLite: 90.72%<br>Recommended edge candidate", REC, op, 11, "strokeWidth=2;")
box(8, 94, 238, 30, "Pruning (50%): 83.49%", WHITE, op, 11)
text(8, 128, 238, 40, "Accuracy on the validation split", op, 10, "fontStyle=3;fontColor=#000000;")
edge(m[-1], ev, mob, f"exitX=1;exitY=0.5;entryX=0;entryY={36/104:.3f};")
edge(ev, op, mob, f"exitX=1;exitY={36/104:.3f};entryX=0;entryY={36/172:.3f};")
gc = box(104, 146, 450, 78, "Grad-CAM<br>explainability<br>(melanoma)", INNER, mob, 11,
         "align=left;spacingLeft=10;")
img(190, 6, 66, 66, MEL, gc)
img(330, 6, 66, 66, OVL, gc)
text(262, 24, 62, 30, "→", gc, 18)
edge(m[1], gc, mob, f"dashed=1;dashPattern=3 3;exitX=0.5;exitY=1;entryX={37/450:.3f};entryY=0;")

text(LX, TOP + 241, LW, 20, "Independent branches: no ensemble or prediction fusion", fs=10,
     extra="fontStyle=3;fontColor=#000000;")

dino, d = lane(TOP + 268, 180, "Branch 2: DINOv2 (Vision Transformer, ~86.6M parameters)", DINO, [
    "Input<br>224 × 224", "Patch embed<br>256 + CLS", "Transformer<br>encoder<br>12 blocks",
    "CLS token", "Linear head<br>768 → 3", "Softmax"])
ev2 = card(dino, 584, 50, 180, 104, "DINOv2 FP32", [
    "Accuracy: 86.66%", "Macro F1: 86.82%", "Size: 330.34 MiB", "CPU latency: 373.35 ms"])
op2 = box(794, 50, 254, 98, "", INNER, dino)
box(8, 8, 117, 30, "Dynamic INT8:<br>61.59%", WHITE, op2, 10)
box(129, 8, 117, 30, "Folded INT8:<br>61.59%", WHITE, op2, 10)
box(8, 46, 238, 44, "INT8 not recommended<br>(significant accuracy drop)", BAD, op2, 11, "strokeWidth=2;")
edge(d[-1], ev2, dino, f"exitX=1;exitY=0.5;entryX=0;entryY={36/104:.3f};")
edge(ev2, op2, dino, f"exitX=1;exitY={36/104:.3f};entryX=0;entryY={36/98:.3f};")
edge(sp, mob, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;jettySize=auto;")
edge(sp, dino, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# ---- Output ----
out = group(1438, TOP, 150, 448, "Classification<br>Output", WHITE)
for i, (t, p) in enumerate([("BCC (Class 0)", BCC), ("Melanoma (Class 1)", MEL), ("Nevus (Class 2)", NEV)]):
    box(10, 56 + i * 88, 130, 80, t, INNER, out, 11, "verticalAlign=bottom;spacingBottom=2;")
    img(51, 61 + i * 88, 48, 48, p, out)
box(10, 324, 130, 98, "Raspberry Pi 4<br>INT8: 13.9 ms", INNER, out, 10,
    "verticalAlign=bottom;spacingBottom=3;")
img(40, 331, 70, 45, RPI, out, square=False, border="none")
edge(mob, out, extra=f"exitX=1;exitY=0.5;entryX=0;entryY={116/448:.3f};")
edge(dino, out, extra=f"exitX=1;exitY=0.5;entryX=0;entryY={358/448:.3f};")

xml = ('<mxfile host="drawio"><diagram id="dv" name="DermaVision Architecture">'
       '<mxGraphModel dx="1480" dy="500" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
       'fold="1" page="1" pageScale="1" pageWidth="1610" pageHeight="520" background="#FFFFFF" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
open(OUT, "w", encoding="utf-8").write(xml)

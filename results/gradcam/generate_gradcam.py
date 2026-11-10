"""
Generate reproducible MobileNetV2 FP32 Grad-CAM visualizations for DermaVision.

Uses the verified checkpoint 'models/best_mobilenetv2.keras' strictly in read-only mode
and two-stage nested sub-model replay via gradcam.py to extract activations and gradients
from 'mobilenetv2_1.00_224/out_relu' (shape: 7x7x1280).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["KERAS_HOME"] = str(ROOT / ".keras_cache")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from gradcam import (
    preprocess_image_tf,
    make_gradcam_heatmap,
    upsample_heatmap,
    colorize_heatmap,
    overlay_gradcam,
)

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
IDX_TO_CLASS = {0: "BCC", 1: "Melanoma", 2: "Nevus"}

KERAS_MODEL_REL = "models/best_mobilenetv2.keras"
INT8_MODEL_REL = "models/best_mobilenetv2_full_int8.tflite"
TARGET_LAYER = "out_relu"
COLORMAP_NAME = "jet"
OVERLAY_ALPHA = 0.40

SELECTED_SPECS = [
    {
        "sample_key": "01_bcc_tp_ISIC_0073082",
        "role_label": "Case 1 — BCC (Correct)",
        "outcome_status": "True Positive",
        "image_id": "ISIC_0073082",
    },
    {
        "sample_key": "02_melanoma_tp_ISIC_0033522",
        "role_label": "Case 2 — Melanoma (Correct)",
        "outcome_status": "True Positive",
        "image_id": "ISIC_0033522",
    },
    {
        "sample_key": "03_nevus_tp_ISIC_0026951",
        "role_label": "Case 3 — Nevus (Correct)",
        "outcome_status": "True Positive",
        "image_id": "ISIC_0026951",
    },
    {
        "sample_key": "04_melanoma_fn_as_nevus_ISIC_0025313",
        "role_label": "Case 4 — Melanoma misclassified as Nevus",
        "outcome_status": "False Negative (Melanoma -> Nevus)",
        "image_id": "ISIC_0025313",
    },
]


def load_deterministic_split():
    """Reconstruct exact 9,956 / 2,489 stratified split and join ISIC 2019 metadata."""
    csv_path = ROOT / "ISIC_2019_Training_GroundTruth.csv"
    img_dir = ROOT / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    if not img_dir.exists():
        img_dir = ROOT / "ISIC_2019_Training_Input"

    df = pd.read_csv(csv_path)
    mel_df = df[df["MEL"] == 1.0].copy()
    mel_df["class"] = "Melanoma"
    nv_df = df[df["NV"] == 1.0].head(4600).copy()
    nv_df["class"] = "Nevus"
    bcc_df = df[df["BCC"] == 1.0].copy()
    bcc_df["class"] = "BCC"

    selected_df = pd.concat([mel_df, nv_df, bcc_df], ignore_index=True)
    data = selected_df[["image", "class"]].copy()
    data["label"] = data["class"].map(CLASS_TO_IDX)

    train_df, val_df = train_test_split(
        data, test_size=0.20, stratify=data["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    val_df["image_path"] = val_df["image"].apply(lambda x: str(img_dir / f"{x}.jpg"))

    meta = pd.read_csv(ROOT / "ISIC_2019_Training_Metadata.csv")
    train_meta = train_df.merge(meta, on="image", how="left")
    val_meta = val_df.merge(meta, on="image", how="left")
    return train_meta, val_meta


def run_int8_check(interpreter, img_batch_np):
    """Read-only cross-check against existing Full INT8 TFLite model."""
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale = float(in_det["quantization_parameters"]["scales"][0])
    in_zp = int(in_det["quantization_parameters"]["zero_points"][0])
    out_scale = float(out_det["quantization_parameters"]["scales"][0])
    out_zp = int(out_det["quantization_parameters"]["zero_points"][0])

    q_in = np.clip(np.round(img_batch_np / in_scale + in_zp), -128, 127).astype(np.int8)
    interpreter.set_tensor(in_det["index"], q_in)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_det["index"])[0]
    p_int8 = (q_out.astype(np.float32) - out_zp) * out_scale
    if p_int8.sum() > 0:
        p_int8 = p_int8 / p_int8.sum()
    idx = int(np.argmax(p_int8))
    return CLASS_NAMES[idx], float(p_int8[idx]) * 100.0


def save_single_sample_panel(rec, orig_pil, heat_pil, over_pil, out_path):
    """Render an annotated 3-panel figure with colorbar for an individual sample."""
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.7), facecolor="white", gridspec_kw={"wspace": 0.08}
    )
    fig.subplots_adjust(left=0.025, right=0.905, top=0.77, bottom=0.07)

    panels = [
        (orig_pil, "Original Dermoscopic Image"),
        (heat_pil, f"Grad-CAM Heatmap ({TARGET_LAYER}, 7x7 -> {rec['original_width']}x{rec['original_height']})"),
        (over_pil, f"Heatmap Overlay (Jet, alpha = {OVERLAY_ALPHA:.2f})"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#0F172A", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#94A3B8")
            sp.set_linewidth(0.9)

    cax = fig.add_axes([0.922, 0.09, 0.018, 0.66])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=mpl.colormaps[COLORMAP_NAME])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized Activation Importance [0, 1]", fontsize=9, color="#1E293B")
    cbar.ax.tick_params(labelsize=8.5)

    status_color = "#15803D" if rec["outcome_status"] == "True Positive" else "#B91C1C"
    fig.text(
        0.025, 0.935,
        f"{rec['role_label']}   |   File: {rec['image_filename']}   |   Lesion ID: {rec['lesion_id']} (Train Overlap: {rec['train_lesion_overlap_count']})",
        fontsize=11.5, fontweight="bold", color="#0F172A",
    )
    fig.text(
        0.025, 0.875,
        f"Ground Truth: {rec['ground_truth']}   |   Predicted (FP32): {rec['predicted_class']} ({rec['fp32_confidence_pct']:.4f}%)   "
        f"[P(BCC)={rec['fp32_prob_bcc']*100:.2f}%, P(Melanoma)={rec['fp32_prob_melanoma']*100:.4f}%, P(Nevus)={rec['fp32_prob_nevus']*100:.4f}%]",
        fontsize=9.5, fontweight="bold", color=status_color,
    )
    fig.text(
        0.025, 0.822,
        f"Checkpoint: {rec['model_checkpoint']}   |   Target Layer: {rec['target_layer']} ({rec['feature_map_shape']})   "
        f"|   INT8 Check: {rec['full_int8_predicted_class']} ({rec['full_int8_confidence_pct']:.2f}%)",
        fontsize=8.7, color="#475569",
    )
    fig.text(
        0.025, 0.018,
        "Note: Post-hoc Grad-CAM attribution aid over coarse 7x7 spatial features; does not prove clinical explainability or diagnostic validity.",
        fontsize=8.2, color="#64748B", style="italic",
    )
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def save_master_comparison(records, sample_images, out_path):
    """Render the publication-grade 4x3 multi-sample Grad-CAM comparison grid."""
    fig, axes = plt.subplots(
        4, 3, figsize=(13.6, 16.0), facecolor="white",
        gridspec_kw={"hspace": 0.30, "wspace": 0.06}
    )
    fig.subplots_adjust(left=0.032, right=0.885, top=0.882, bottom=0.052)

    col_headers = [
        "Original Dermoscopic Image",
        "Grad-CAM Heatmap (7x7 -> Original)",
        f"Heatmap Overlay (Jet, alpha = {OVERLAY_ALPHA:.2f})",
    ]
    for c_idx, c_title in enumerate(col_headers):
        axes[0, c_idx].set_title(c_title, fontsize=11.0, fontweight="bold", color="#0F172A", pad=26)

    for r_idx, (rec, imgs) in enumerate(zip(records, sample_images)):
        is_tp = rec["outcome_status"] == "True Positive"
        accent = "#15803D" if is_tp else "#B91C1C"
        tag_text = "TP" if is_tp else "FN"
        conf_str = f"{rec['fp32_confidence_pct']:.4f}%" if rec["fp32_confidence_pct"] < 99.99995 else "100.00%"

        ax_left = axes[r_idx, 0]
        pos = ax_left.get_position()
        row_top_y = pos.y1 + 0.007
        fig.text(
            pos.x0,
            row_top_y,
            (
                f"{rec['role_label']} [{tag_text}]   |   {rec['image_filename']}   |   "
                f"Lesion: {rec['lesion_id']} (Non-Overlapping Val, 0 Train Match)   |   "
                f"GT: {rec['ground_truth']} -> Pred: {rec['predicted_class']} ({conf_str})"
            ),
            fontsize=8.45,
            fontweight="bold",
            color=accent,
            va="bottom",
            ha="left",
        )

        triplets = [imgs["orig"], imgs["heat"], imgs["over"]]
        for c_idx, im in enumerate(triplets):
            ax = axes[r_idx, c_idx]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(accent if c_idx == 2 else "#94A3B8")
                sp.set_linewidth(1.5 if (c_idx == 2 and not is_tp) else 0.9)

        axes[r_idx, 1].text(
            0.03, 0.04,
            f"mean={rec['heatmap_mean']:.2f} | >0: {rec['heatmap_nonzero_7x7_cells']}/49",
            transform=axes[r_idx, 1].transAxes,
            fontsize=8.0, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F172A", alpha=0.72, edgecolor="none"),
        )
        axes[r_idx, 2].text(
            0.03, 0.04,
            f"P(BCC)={rec['fp32_prob_bcc']*100:.1f}% | P(MEL)={rec['fp32_prob_melanoma']*100:.1f}% | P(NV)={rec['fp32_prob_nevus']*100:.1f}%",
            transform=axes[r_idx, 2].transAxes,
            fontsize=7.7, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F172A", alpha=0.75, edgecolor="none"),
        )

    cax = fig.add_axes([0.908, 0.07, 0.019, 0.79])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=mpl.colormaps[COLORMAP_NAME])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized Grad-CAM Importance [0.0 = Low, 1.0 = Peak]", fontsize=10.2, fontweight="bold", color="#0F172A", labelpad=10)
    cbar.ax.tick_params(labelsize=9.2)

    fig.suptitle(
        "DermaVision — MobileNetV2 FP32 Grad-CAM Spatial Attribution Analysis",
        fontsize=14.8, fontweight="bold", color="#0F172A", y=0.968,
    )
    fig.text(
        0.48, 0.937,
        (
            f"Model Checkpoint: {KERAS_MODEL_REL}  |  Target Layer: mobilenetv2_1.00_224/{TARGET_LAYER} (7x7x1280) via Two-Stage Nested Replay  |  "
            f"Pre-Softmax Target Score y^c\n"
            f"Sample Provenance: Clean lesion-ID non-overlapping subset (n = 1,144 of 2,489 val images, ISIC 2019)  |  "
            f"Fixed Color Scale: Jet [0, 1], Bicubic Scalar Upsampling, alpha = {OVERLAY_ALPHA:.2f}"
        ),
        fontsize=9.1, color="#334155", ha="center", va="center", linespacing=1.45,
    )
    fig.text(
        0.48, 0.022,
        (
            "Limitation & Scientific Disclaimer: Grad-CAM visualizes coarse (7x7) spatial gradients highlighting image regions influencing MobileNetV2's predicted score;\n"
            "it is a post-hoc inspection aid only and does NOT prove clinical explainability or model correctness (Case 4 focuses on the lesion despite Melanoma -> Nevus error)."
        ),
        fontsize=8.4, color="#475569", ha="center", va="center", style="italic", linespacing=1.35,
    )

    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    out_dir = ROOT / "results" / "gradcam"
    samples_dir = out_dir / "gradcam_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_meta, val_meta = load_deterministic_split()
    train_known_lesions = set(train_meta["lesion_id"].dropna().astype(str))

    print(f"Loading FP32 model (read-only): {KERAS_MODEL_REL}")
    keras_model = tf.keras.models.load_model(str(ROOT / KERAS_MODEL_REL), compile=False)

    print(f"Loading Full INT8 TFLite model (read-only cross-check): {INT8_MODEL_REL}")
    int8_interp = tf.lite.Interpreter(model_path=str(ROOT / INT8_MODEL_REL))
    int8_interp.allocate_tensors()

    records = []
    sample_images = []

    for spec in SELECTED_SPECS:
        img_id = spec["image_id"]
        matches = val_meta[val_meta["image"] == img_id]
        if len(matches) != 1:
            raise RuntimeError(f"Expected 1 validation match for {img_id}, found {len(matches)}")
        row = matches.iloc[0]

        lesion_id = str(row["lesion_id"]) if pd.notna(row["lesion_id"]) else "N/A"
        overlap_count = int((train_meta["lesion_id"].astype(str) == lesion_id).sum()) if pd.notna(row["lesion_id"]) else 0
        if lesion_id in train_known_lesions:
            raise RuntimeError(f"Selected sample {img_id} overlaps training lesion {lesion_id}!")

        img_path = Path(row["image_path"])
        orig_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_pil.size

        img_tensor = preprocess_image_tf(img_path, target_size=(224, 224))

        # Exact direct model forward pass check to guarantee replay matches model(img_tensor) identically
        direct_probs = keras_model(img_tensor, training=False).numpy()[0]

        heatmap_7x7, details = make_gradcam_heatmap(
            img_tensor,
            keras_model,
            last_conv_layer_name=TARGET_LAYER,
            pred_index=None,
            use_pre_softmax=True,
            return_details=True,
        )
        max_prob_diff = float(np.max(np.abs(direct_probs - details["probs"])))
        if max_prob_diff > 1e-5:
            raise RuntimeError(f"Replay probability mismatch for {img_id}: {max_prob_diff}")

        pred_idx = details["pred_index"]
        pred_class = IDX_TO_CLASS[pred_idx]
        probs = details["probs"]
        logits = details["logits"]
        int8_pred, int8_conf = run_int8_check(int8_interp, img_tensor.numpy())

        heat_pil = colorize_heatmap(heatmap_7x7, colormap=COLORMAP_NAME, target_size=(orig_w, orig_h))
        over_pil = overlay_gradcam(img_path, heatmap_7x7, alpha=OVERLAY_ALPHA, colormap=COLORMAP_NAME)

        prefix = spec["sample_key"]
        orig_out = samples_dir / f"{prefix}_original.png"
        heat_out = samples_dir / f"{prefix}_heatmap.png"
        over_out = samples_dir / f"{prefix}_overlay.png"
        panel_out = samples_dir / f"{prefix}_panel.png"

        orig_pil.save(orig_out)
        heat_pil.save(heat_out)
        over_pil.save(over_out)

        rec = {
            "sample_key": prefix,
            "role_label": spec["role_label"],
            "image_id": img_id,
            "image_filename": f"{img_id}.jpg",
            "lesion_id": lesion_id,
            "train_lesion_overlap_count": overlap_count,
            "validation_subset": "Clean lesion-ID non-overlapping val slice (n=1,144 of 2,489)",
            "ground_truth": row["class"],
            "predicted_class": pred_class,
            "outcome_status": spec["outcome_status"],
            "fp32_confidence_pct": round(float(probs[pred_idx]) * 100.0, 6),
            "fp32_prob_bcc": float(probs[0]),
            "fp32_prob_melanoma": float(probs[1]),
            "fp32_prob_nevus": float(probs[2]),
            "fp32_target_logit": float(logits[pred_idx]),
            "replay_max_prob_diff": max_prob_diff,
            "full_int8_predicted_class": int8_pred,
            "full_int8_confidence_pct": round(int8_conf, 4),
            "model_checkpoint": KERAS_MODEL_REL,
            "target_layer": f"mobilenetv2_1.00_224/{TARGET_LAYER}",
            "feature_map_shape": "x".join(str(x) for x in details["feature_map_shape"][1:]),
            "grad_max_abs": details["grad_max_abs"],
            "heatmap_min": float(heatmap_7x7.min()),
            "heatmap_mean": float(heatmap_7x7.mean()),
            "heatmap_max": float(heatmap_7x7.max()),
            "heatmap_nonzero_7x7_cells": int((heatmap_7x7 > 0).sum()),
            "original_width": orig_w,
            "original_height": orig_h,
            "colormap": COLORMAP_NAME,
            "overlay_alpha": OVERLAY_ALPHA,
            "anatom_site_general": row["anatom_site_general"] if pd.notna(row["anatom_site_general"]) else "N/A",
            "age_approx": row["age_approx"] if pd.notna(row["age_approx"]) else "N/A",
            "sex": row["sex"] if pd.notna(row["sex"]) else "N/A",
            "original_image_path": orig_out.relative_to(ROOT).as_posix(),
            "heatmap_image_path": heat_out.relative_to(ROOT).as_posix(),
            "overlay_image_path": over_out.relative_to(ROOT).as_posix(),
            "panel_image_path": panel_out.relative_to(ROOT).as_posix(),
        }
        save_single_sample_panel(rec, orig_pil, heat_pil, over_pil, panel_out)
        records.append(rec)
        sample_images.append({"orig": orig_pil, "heat": heat_pil, "over": over_pil})
        print(
            f"Generated {prefix}: GT={rec['ground_truth']} | Pred={pred_class} ({rec['fp32_confidence_pct']:.4f}%) "
            f"| INT8={int8_pred} | hm_mean={rec['heatmap_mean']:.3f} | replay_diff={max_prob_diff:.2e}"
        )

    comp_path = out_dir / "gradcam_comparison.png"
    save_master_comparison(records, sample_images, comp_path)

    csv_path = out_dir / "gradcam_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    print(f"\nSaved comparison figure: {comp_path}")
    print(f"Saved sample directory:   {samples_dir}")
    print(f"Saved results CSV:        {csv_path}")


if __name__ == "__main__":
    main()

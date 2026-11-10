# DermaVision — MobileNetV2 FP32 Grad-CAM Generation Report

**Generated:** `2026-11-10`  
**Verified Checkpoint:** [models/best_mobilenetv2.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) (unmodified; read-only evaluation)  
**Target Activation Layer:** `mobilenetv2_1.00_224/out_relu` (`7×7×1280` final convolutional activation tensor)

---

## 1. Implementation Summary & Prototype Repair

- **Original Backup Preserved:** Exact copy of the pre-existing 62-line prototype saved at [gradcam.py.bak](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py.bak).
- **Updated Script:** [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) now implements two critical fixes while preserving backward-compatible function names ([`make_gradcam_heatmap`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L38-L140), [`overlay_gradcam`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L161-L171)):
  1. **Two-Stage Nested Model Replay (`TensorFlow 2.21.0` / `Keras 3.15.0`):** Because MobileNetV2 (`mobilenetv2_1.00_224`, `154` inner layers) sits inside a nested `Functional` block ahead of `global_average_pooling2d -> dense (256) -> dropout (0.2) -> dense_1 (64) -> dropout_1 (0.1) -> dense_2 (3)`, [`make_gradcam_heatmap`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L38-L140) builds an inner sub-model extracting `(out_relu, inner_output)` and replays the outer classifier layers (`head_layers`) inside `tf.GradientTape()`. Direct replay verification confirmed `replay_max_prob_diff = 0.0` (`< 1e-6`) against `keras_model(img_tensor)` across all selected samples.
  2. **Pre-Softmax Score Differentiation & Calibrated Color Mapping:** Differentiates pre-softmax class logits (`y^c` before softmax via `dense_2.kernel` / `dense_2.bias`) rather than post-softmax probabilities, avoiding float32 softmax gradient vanishing on `~100%`-confident predictions (`|∇_p| <= 1e-12` vs `|∇_logit| ≈ 1.3e-3 – 2.3e-3`). Scalar `[0, 1]` heatmaps are upsampled from `7×7` to original image resolution with `Image.BICUBIC` (clamped `[0, 1]`) and mapped with a fixed `[0.0, 1.0]` Jet scale and fixed blend `alpha = 0.40`.
- **Reproducible Generator Script:** [generate_gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/generate_gradcam.py).

---

## 2. Selected Validation Samples & Predictions

All four examples were drawn from the **strict known-`lesion_id` non-overlapping validation subset** (`train_lesion_overlap_count = 0` across all `9,956` training images, part of the `n = 1,144` non-overlapping slice of the `2,489`-image `ISIC 2019` validation split):

| Case | File (`ISIC 2019`) | Lesion ID (`0` Train Overlap) | Size / Site / Demographics | Ground Truth | FP32 Predicted (`best_mobilenetv2.keras`) | Full INT8 Check | Logit `y^c` / `grad_max_abs` | Heatmap Mean / Active Cells |
|---|---|---|---|---|---|---|---|---|
| **Case 1 (BCC TP)** | `ISIC_0073082.jpg` | `BCN_0002149` | `1024×1024`, `anterior torso`, `80y M` | `BCC` | **`BCC` (`100.00%`)** (`P(BCC)=1.0`, `P(MEL)=6.4e-9`, `P(NV)=2.1e-12`) | `BCC` (`100.00%`) | `17.90` / `2.31e-3` | `0.398` / `49/49` |
| **Case 2 (MEL TP)** | `ISIC_0033522.jpg` | `HAM_0004321` | `600×450`, `posterior torso`, `40y F` | `Melanoma` | **`Melanoma` (`100.00%`)** (`P(BCC)=3.8e-10`, `P(MEL)=1.0`, `P(NV)=3.5e-11`) | `Melanoma` (`100.00%`) | `14.67` / `1.83e-3` | `0.402` / `49/49` |
| **Case 3 (NV TP)** | `ISIC_0026951.jpg` | `HAM_0003861` | `600×450`, `posterior torso`, `40y F` | `Nevus` | **`Nevus` (`100.00%`)** (`P(BCC)=1.1e-16`, `P(MEL)=4.6e-13`, `P(NV)=1.0`) | `Nevus` (`100.00%`) | `17.07` / `1.41e-3` | `0.440` / `49/49` |
| **Case 4 (MEL→NV FN)** | `ISIC_0025313.jpg` | `HAM_0001378` | `600×450`, `lower extremity`, `35y F` | `Melanoma` | **`Nevus` (`99.9999%`)** (`P(BCC)=2.2e-9`, `P(MEL)=9.84e-7`, `P(NV)=0.999999`) | `Nevus` (`100.00%`) | `8.99` / `1.32e-3` | `0.304` / `49/49` |

---

## 3. Visual Inspection & Interpretation Boundary

Visual inspection of [gradcam_comparison.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_comparison.png) and all per-sample cards in `results/gradcam/gradcam_samples/` confirms clean geometric alignment, legible metadata, zero clipped text, and calibrated `[0.0, 1.0]` Jet colorbars:

- **Correct Predictions (Cases 1–3):** Peak activation concentrates over the central pearly BCC nodule and arborizing vessels (`ISIC_0073082.jpg`), the asymmetric pigmented Melanoma core (`ISIC_0033522.jpg`), and the circumscribed Nevus body/upper margin (`ISIC_0026951.jpg`, accompanied by moderate low-to-mid background cyan tint `mean = 0.44`).
- **False Negative Diagnostic Insight (Case 4 — `ISIC_0025313.jpg`):** Peak activation clearly centers directly on the dense dark lesion body (`Nevus` predicted at `99.9999%` confidence even though ground truth is `Melanoma`). **This demonstrates an essential limitation:** coarse `7×7` Grad-CAM localization inside the lesion boundary **does not prove clinical explainability, histopathological validity, or diagnostic correctness** — a CNN can attend cleanly to the lesion itself and still misclassify Melanoma as Nevus.

---

## 4. Exact Generated Output Paths

- **Master Comparison Figure:** [gradcam_comparison.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_comparison.png)
- **Quantitative Results Table:** [gradcam_results.csv](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_results.csv)
- **Generator Script:** [generate_gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/generate_gradcam.py)
- **Updated Utility & Original Backup:** [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py), [gradcam.py.bak](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py.bak)
- **Per-Sample Outputs (`results/gradcam/gradcam_samples/`):**
  - Case 1 (`BCC` TP): [original](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/01_bcc_tp_ISIC_0073082_original.png), [heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/01_bcc_tp_ISIC_0073082_heatmap.png), [overlay](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/01_bcc_tp_ISIC_0073082_overlay.png), [panel](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/01_bcc_tp_ISIC_0073082_panel.png)
  - Case 2 (`Melanoma` TP): [original](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/02_melanoma_tp_ISIC_0033522_original.png), [heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/02_melanoma_tp_ISIC_0033522_heatmap.png), [overlay](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/02_melanoma_tp_ISIC_0033522_overlay.png), [panel](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/02_melanoma_tp_ISIC_0033522_panel.png)
  - Case 3 (`Nevus` TP): [original](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/03_nevus_tp_ISIC_0026951_original.png), [heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/03_nevus_tp_ISIC_0026951_heatmap.png), [overlay](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/03_nevus_tp_ISIC_0026951_overlay.png), [panel](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/03_nevus_tp_ISIC_0026951_panel.png)
  - Case 4 (`Melanoma -> Nevus` FN): [original](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/04_melanoma_fn_as_nevus_ISIC_0025313_original.png), [heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/04_melanoma_fn_as_nevus_ISIC_0025313_heatmap.png), [overlay](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/04_melanoma_fn_as_nevus_ISIC_0025313_overlay.png), [panel](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_samples/04_melanoma_fn_as_nevus_ISIC_0025313_panel.png)


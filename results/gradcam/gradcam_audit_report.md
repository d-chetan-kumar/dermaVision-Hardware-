# DermaVision — Read-Only Grad-CAM & Explainability Audit Report

**Audit Date:** `2026-11-10`  
**Audit Mode:** Strictly read-only inspection (no model training, no checkpoint edits, no image generation)  
**Overall Readiness Verdict:** **NOT PAPER-READY — Zero Grad-CAM or explainability visualizations currently exist in the workspace or manuscripts.**

---

## 1. Executive Summary

A full repository, asset, manuscript, and runtime compatibility audit confirms that DermaVision **does not yet possess any generated Grad-CAM visualizations** for the research paper (corresponding to Reviewer 3 Comment `R3.2` in [reviewer_comment_response_matrix.csv](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/reviewer_comment_response_matrix.csv#L15)):

1. **Only one unintegrated utility script exists:** [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) (`62` lines, `2,006` B, untracked in Git, never imported or called anywhere in the codebase).
2. **Zero generated Grad-CAM artifacts exist:** There are `0` Grad-CAM heatmaps, `0` heatmap overlays, `0` side-by-side comparison figures, and `0` prior Grad-CAM evaluation reports across `graphs/`, `metrics/`, `results/`, root, or the LaTeX/PDF manuscripts.
3. **MobileNetV2 Grad-CAM currently fails out-of-the-box:** Testing [make_gradcam_heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L6-L32) in read-only RAM against [best_mobilenetv2.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) (`TensorFlow 2.21.0` / `Keras 3.15.0`) throws `ValueError` for every valid convolutional layer (`Conv_1`, `out_relu`) because the backbone sits inside a nested `Functional` sub-model (`mobilenetv2_1.00_224`), and throws `ValueError: Output with path 0 is not connected to inputs` even when targeting `mobilenetv2_1.00_224` directly.
4. **DINOv2 has zero explainability implementation:** [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) supports TensorFlow/Keras only; there is no PyTorch hook, patch-token Grad-CAM, or Vision Transformer attention rollout for [dinov2_100_epoch_best.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth).

---

## 2. Inventory of Existing Grad-CAM Artifacts & Paths

| Artifact Category | Status | Exact Workspace Paths | Audit Findings |
|---|---|---|---|
| **Grad-CAM Implementation Scripts** | **1 Unused Prototype** | [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) | Contains only [make_gradcam_heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L6-L32) and [overlay_gradcam](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L34-L62). Modified `2026-08-31T18:29:58`; zero imports or callers in [streamlit_app.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py), [prediction.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction.py), [prediction_int8.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py), [evaluation.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py), or [test_pipeline.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/test_pipeline.py). |
| **Generated Grad-CAM Heatmaps** | **None (`0`)** | `NONE` | No standalone spatial activation maps exist anywhere in `results/`, `metrics/`, `graphs/`, or root. (`sns.heatmap(...)` calls in [evaluation.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py#L286), [evaluate_final_ensemble.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py#L278), and [build_kaggle_notebook.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/build_kaggle_notebook.py#L613) plot confusion matrices only.) |
| **Original Images with Heatmap Overlays** | **None (`0`)** | `NONE` | No Grad-CAM blended dermoscopic images exist. |
| **Side-by-Side Comparison Visualizations** | **None (`0`)** | `NONE` | Two unlinked image grids in root—[melanoma_candidates.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/melanoma_candidates.png) (`800×600`) and [predictions_grid.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/predictions_grid.png) (`908×604`)—show raw lesion thumbnails only, with **zero heatmaps, zero class predictions, and zero colorbars**. |
| **Grad-CAM Reports / Manuscript Figures** | **None (`0` prior)** | [reviewer_comment_response_matrix.csv](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/reviewer_comment_response_matrix.csv#L15)<br>[ICADCML_2027_DermaVision_updated.tex](file:///C:/Users/HP/Downloads/ICADCML_2027_DermaVision_updated.tex) | Matrix row `R3.2` explicitly marks Grad-CAM as `Not Addressed`. Zero Grad-CAM figures appear in [ICADCML_2027_DermaVision_updated.tex](file:///C:/Users/HP/Downloads/ICADCML_2027_DermaVision_updated.tex) or the associated PDF manuscripts (`Dermavision_Manuscript.pdf`, `ISIC_Dermavison_manuscript (1).pdf`, `1861_CR.pdf`). |

---

## 3. Model Compatibility Verification

### 3.1 Read-Only Runtime Probe Results

A read-only in-memory probe (using dummy zero tensors only; no dataset images read and no images saved) verified actual runtime compatibility across all DermaVision checkpoints:

| Architecture / Artifact | Exact Checkpoint Path | Grad-CAM Implemented? | Technical Compatibility & Verified Findings |
|---|---|---|---|
| **MobileNetV2 FP32 Original** (`30.21 MiB`) | [models/best_mobilenetv2.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) | **Broken Prototype** ([gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py)) | **Fails as written** in `TF 2.21.0` / `Keras 3.15.0`. Outer layers are `['input_layer_1', 'mobilenetv2_1.00_224', 'global_average_pooling2d', 'dense', 'dropout', 'dense_1', 'dropout_1', 'dense_2']`. Inner `mobilenetv2_1.00_224` (`154` layers) holds `Conv_1` and `out_relu` `(1, 7, 7, 1280)`. Calling `make_gradcam_heatmap(..., 'Conv_1')` or `'out_relu'` raises `ValueError: No such layer`; calling `'mobilenetv2_1.00_224'` or passing `inner.get_layer('Conv_1').output` raises `ValueError: Output with path 0 is not connected to inputs`. **Fixable:** a two-stage nested replay (`inner_sub` -> head `model.layers[2:]` inside `tf.GradientTape`) produced valid `(1, 7, 7, 1280)` activations and non-`None` gradients. |
| **MobileNetV2 50% Pruned** (`30.21 MiB`) | [models/best_mobilenetv2_pruned.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_pruned.keras) | **Not Used / Same Bug** | Identical 8-layer nested topology; fails identically under [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) and succeeds only via two-stage replay. Not recommended for deployment (`73.45%` Melanoma recall). |
| **MobileNetV2 Dynamic TFLite** (`2.74 MiB`) | [models/best_mobilenetv2.tflite](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.tflite) | **Incompatible** | Used by [prediction.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction.py) / [streamlit_app.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py). TFLite flatbuffer `Interpreter` exposes neither gradients nor intermediate feature maps for backward Grad-CAM. |
| **MobileNetV2 Full INT8 TFLite** (`2.94 MiB`, Recommended Edge Candidate) | [models/best_mobilenetv2_full_int8.tflite](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) | **Incompatible** | Used by [prediction_int8.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py). Same TFLite limitation: INT8 TFLite runtimes have no autodiff/backprop graph. Any MobileNetV2 Grad-CAM visual must be computed from the source FP32 Keras model ([best_mobilenetv2.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras)) with provenance explicitly disclosed (and prediction agreement checked against INT8 on displayed samples). |
| **DINOv2 ViT-B/14 FP32** (`330.34 MiB` total) | [models/dinov2_100_epoch_best.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) + cached `dinov2_vitb14_pretrain.pth` | **Not Implemented (`0%`)** | PyTorch `DinoVisionTransformer` (`12` blocks, patch size `14`, `0` registers, `257` tokens at `224×224`). The linear probe [CombinedDINOv2](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_dynamic_int8_inference.py#L46-L56) classifies strictly from `x_norm_clstoken` `(1, 768)` via `nn.Linear(768, 3)`. There is **no post-backbone 2D conv layer** for CNN Grad-CAM; explainability requires PyTorch self-attention extraction (e.g., last-block CLS->patch attention or multi-block Attention Rollout over the `256` `16×16` patch tokens) or gradient-weighted patch attention, none of which exist in the project. |
| **DINOv2 Native Dynamic INT8** (`87.38 MiB`) | [models/dinov2/dinov2_vitb14_dynamic_int8.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_vitb14_dynamic_int8.pth) | **Not Implemented & Rejected** | PyTorch quantized `DynamicQuantizedLinear` modules; unsuitable clinically due to `-25.07 pp` accuracy collapse (`61.59%`). |

**Checkpoints Used in Existing Grad-CAM Visualizations:** **None.** Zero Grad-CAM outputs have been produced from any checkpoint.

---

## 4. Visualization Quality & Code Inspection

### 4.1 Generated Image Inspection
- **Grad-CAM heatmaps & overlays:** **None present** (`0` files). Overlay alignment, predicted vs. ground-truth accuracy, colorbar scales, and per-panel typography cannot pass because no Grad-CAM figures exist.
- **Unrelated root image grids inspected:**
  - [melanoma_candidates.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/melanoma_candidates.png) (`800×600` RGB, valid non-corrupted PNG): 4×3 raw contact sheet showing 12 lesion tiles overlaid with text `"Black"` (`ISIC_0000002`, `ISIC_0000022_downsampled`, `ISIC_0000026_downsampled`, `ISIC_0000030_downsampled`, `ISIC_0000031_downsampled`, `ISIC_0000035_downsampled`, `ISIC_0000040_downsampled`, `ISIC_0000046_downsampled`, `ISIC_0000049_downsampled`, `ISIC_0000054_downsampled`, `ISIC_0000056_downsampled`, `ISIC_0000074`). No heatmaps, predictions, or colorbars.
  - [predictions_grid.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/predictions_grid.png) (`908×604` RGB, valid non-corrupted PNG): 3×2 raw dermoscopic image grid with black borders. No heatmaps, filenames, class predictions, ground-truth labels, or colorbars.

### 4.2 Static Code Defects in [gradcam.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py)
Beyond the Keras 3 nested-submodel crash documented in Section 3.1, [overlay_gradcam](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L34-L62) has multiple publication-readiness gaps:
- **Graph bug:** `inputs=[model.inputs]` wraps a list inside a list (`[[KerasTensor]]`), triggering Keras structure mismatches alongside nested sub-model disconnection.
- **Preprocessing omitted:** [make_gradcam_heatmap](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py#L6-L32) expects an already resized & `/255.0` normalized `(1, 224, 224, 3)` float tensor; there is no integrated MobileNetV2 preprocessing helper enforcing consistent alignment between inference and visualization.
- **Map normalization pitfall:** `tf.keras.utils.array_to_img(jet_heatmap)` (`line 53`) defaults to `scale=True`, which independently re-normalizes RGB channels from `[min, max]` to `[0, 255]` per image and can distort low-contrast activations instead of preserving a fixed `[0, 1]` scale tied to a colorbar.
- **Missing publication elements:** Returns an unannotated blended `PIL.Image` using fixed `alpha=0.4` and the rainbow `"jet"` colormap—missing a multi-panel layout (`Original | Heatmap | Overlay`), explicit color intensity scale (`0.0` to `1.0`), ISIC image ID, ground-truth class, predicted class, softmax confidence, and target layer provenance (`out_relu`, `7×7`).

---

## 5. Sample Selection Audit (`BCC`, `MEL`, `NV`)

| Class (`class_names.json` index) | Grad-CAM Samples Present | Verified Image IDs with Grad-CAM | Prediction / Ground-Truth Outcomes |
|---|---:|---|---|
| **BCC** (`0`) | **0** | `None` | `Not evaluated for Grad-CAM` |
| **Melanoma (`MEL`)** (`1`) | **0** | `None` *(Note: [melanoma_candidates.png](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/melanoma_candidates.png) shows 12 unpredicted raw lesions only)* | `Not evaluated for Grad-CAM` |
| **Nevus (`NV`)** (`2`) | **0** | `None` | `Not evaluated for Grad-CAM` |

No class-balanced correct predictions (true positives) or clinically critical error cases (e.g., `Melanoma -> Nevus` false negatives or `Nevus -> Melanoma` false positives) have been generated or verified with spatial attribution maps.

> [!CAUTION]
> **Explainability Claim Boundary:** Even once generated, Grad-CAM heatmaps and ViT attention maps are post-hoc spatial attribution approximations (especially coarse `7×7` maps from MobileNetV2's final block or `16×16` patch maps from ViT-B/14). They serve only as qualitative visual inspection aids and **do not prove clinical explainability, causal diagnostic reasoning, or dermatological safety**.

---

## 6. Missing Components Required for Paper Readiness

**Conclusion:** **Paper-ready Grad-CAM visualizations DO NOT currently exist.** Before responding to Reviewer 3 (`R3.2`) or inserting qualitative explainability figures into [ICADCML_2027_DermaVision_updated.tex](file:///C:/Users/HP/Downloads/ICADCML_2027_DermaVision_updated.tex), the following components must be implemented under a dedicated `results/gradcam/` pipeline (without altering model checkpoints):

1. **Fix MobileNetV2 Nested Grad-CAM Pipeline:** Implement two-stage nested sub-model gradient extraction targeting `out_relu` `(7×7×1280)` inside `mobilenetv2_1.00_224` on [best_mobilenetv2.keras](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) with exact `/255.0` `224×224` preprocessing; verify agreement between FP32 MobileNetV2 predictions and Full INT8 TFLite predictions ([best_mobilenetv2_full_int8.tflite](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite)) for every showcased sample.
2. **Add DINOv2 ViT-B/14 Spatial Attribution (If Both Branches Are Shown):** Implement non-destructive PyTorch `CLS`-to-patch attention / Attention Rollout (or gradient-weighted `16×16` patch attribution) for [dinov2_100_epoch_best.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) using ImageNet normalization—never mislabeling ViT attention as CNN Grad-CAM.
3. **Curate Class-Representative & Error Cases from Verified Validation Data:** Select verified validation images (preferring clean non-overlapping lesions from the `n = 1,144` non-overlapping slice where available) spanning:
   - True positive examples for all three classes (**BCC**, **Melanoma**, **Nevus**);
   - At least one failure case (**Melanoma misclassified as Nevus**) to candidly inspect spurious background/ruler/hair/border artifacts rather than cherry-picking only successes.
4. **Publication-Grade Multi-Panel Layout:** Render synchronized high-DPI panels (`Original Image | Raw Spatial Map | Overlay`) with exact ISIC filename, ground-truth label, predicted class, softmax probability, model checkpoint provenance, and a calibrated `[0, 1]` colorbar (`scale=False` on colormap RGB mapping).
5. **Visual & Integrity Sign-Off:** Visually inspect all generated PNGs for alignment, legibility at IEEE column width, absence of clipped labels or duplicate tiles, and accompany the figure in the manuscript with an explicit statement of coarse spatial resolution (`7×7` / `16×16`) and the limitation that attribution maps do not establish clinical validity.


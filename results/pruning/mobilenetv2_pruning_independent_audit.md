# DermaVision: Independent Verification-Only Audit of the MobileNetV2 Pruning Experiment

**Audit Timestamp:** 2026-11-10 10:00 IST  
**Target Workspace:** `c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019 dataset)`  
**Primary Artifacts Audited:**
- `models/best_mobilenetv2.keras`
- `models/best_mobilenetv2_pruned.keras`
- `results/pruning/checkpoints/mobilenetv2_pruned_epoch_01.h5` through `epoch_05.h5`
- `prune_and_train_mobilenetv2.py`
- `evaluate_pruned_mobilenetv2.py`
- `results/pruning/mobilenetv2_pruning_report.md`
- `results/pruning/mobilenetv2_pruning_metrics.json`
- `results/pruning/mobilenetv2_pruning_history.csv`
- Earlier reference reports: `dermavision_original_model_verification_report.md` and `results/int8/mobilenetv2_int8_verification_report.md`

**Audit Mode & Compliance:** Strictly **verification-only**. Zero models were retrained, re-quantized, pruned, overwritten, renamed, or deleted. All pre-existing files remain cryptographically identical before and after audit (Section 9).

---

## Executive Audit Classification Matrix

| Audit Area | Classification | Key Finding Summary |
|---|---|---|
| **1. Complete Architecture of `models/best_mobilenetv2.keras`** | **VERIFIED** | 8 top-level layers (`InputLayer` + `mobilenetv2_1.00_224` [154 inner layers] + `GAP` + `Dense(256, relu)` + `Dropout(0.2)` + `Dense(64, relu)` + `Dropout(0.1)` + `Dense(3, softmax)`), exact total **2,602,563** params (**2,568,451** trainable, **34,112** non-trainable). 100% structural match with `best_mobilenetv2_pruned.keras`. |
| **2. Resolution of `Dense(512)->Dense(128)` vs `Dense(256)->Dense(64)`** | **DISCREPANCY** (in Earlier Report) / **VERIFIED** (in Pruning Report) | `dermavision_original_model_verification_report.md` (Section 2A) erroneously listed `Dense(512)->Dense(128)` (which requires **2,979,907** params), contradicting its own Section 6 (`2,602,563` params). Actual saved weights, `config.json`, parameter arithmetic, and pruning artifacts prove **`Dense(256)->Dense(64)`**. |
| **3. Checkpoint Initialization, Weight Lineage & Provenance** | **VERIFIED** (Lineage & Stripping) / **DISCREPANCY** (Selection Rule) / **UNVERIFIED** (Internal Metadata) | `prune_and_train_mobilenetv2.py` transfers weights from `best_mobilenetv2.keras` (`max_diff = 6.52e-08`), `epoch_05.h5` stripped matches `best_mobilenetv2_pruned.keras` bit-for-bit (`max_diff = 0.0`), and surviving non-zero weights show mean cosine similarity **0.999989** across backbone layers. **Discrepancy:** Epoch 5 (`83.49%` @ `50.00%` sparsity) was selected over Epoch 1 (`90.60%` @ `23.65%` sparsity) via target-sparsity filter (`>=49.0%`, docstring says `>=49.5%`). **Unverified:** `.keras` `metadata.json` stores no parent hash; no written pre-run SHA-256 hash existed before pruning. |
| **4. Class Mapping, Preprocessing & Dataset Parity** | **VERIFIED** | Both models use identical mapping `{0: "BCC", 1: "Melanoma", 2: "Nevus"}`, `(224, 224, 3)` RGB float32 `/ 255.0`, deterministic `train_test_split(..., random_state=42)` (`9,956` train / `2,489` val: BCC=665, Mel=904, Nev=920), and identical path ordering & labels. *(Historical note: Kaggle CSV Ep 50 logged `90.12%`, whereas deterministic saved-checkpoint evaluation yields `90.96%`.)* |
| **5. Independent Accuracy Verification (2,489 Val Images)** | **VERIFIED** | Independent evaluation (`TF_ENABLE_ONEDNN_OPTS=0`, `batch=64`) reproduces **90.9602%** (`2,264/2,489`) original and **83.4873%** (`2,078/2,489`) pruned exactly (`-7.47 pp` delta). Stripped Epoch 1 reproduces **90.5986%** (`2,255/2,489`). |
| **6. Per-Class Metrics, Confusion Matrices & Sparsity** | **VERIFIED** | Every macro/per-class P/R/F1 value and confusion matrix cell matches identically. Melanoma recall dropped **-13.61 pp** (`87.06% -> 73.45%`, with `176` melanomas misclassified as Nevus and `64` as BCC). Prunable sparsity verified at **49.9971%** (`1,234,825 / 2,469,792` zeros); overall **47.4465%**; `gzip -9` drops `28.23 MB -> 6.01 MB`. |
| **7. Latency Benchmark Methodology & Hardware Scope** | **VERIFIED** (Timing Scope) / **UNVERIFIED** (SKU/System State) | Both INT8 & Pruning scripts use `batch=1`, `50` warm-up + `200` timed iterations on `val_images_fp32[:200]` via `time.perf_counter()`, **excluding** model loading and JPEG decode/resize/`/255.0`. Only Full INT8 includes numpy quantize+dequantize inside the timed loop. Exact Intel marketing SKU & background Windows state during earlier runs are unverified post hoc. |
| **8. Cross-Report Benchmark Comparison (INT8 vs Pruning)** | **MAJOR DISCREPANCY** | INT8 report (`08:18:17`) ran with `TF_ENABLE_ONEDNN_OPTS="0"` (`269.56 ms` FP32 / `16.35 ms` INT8); Pruning report (`09:29:42`) ran with `"1"` (`131.77 ms` FP32 / `6.63 ms` INT8 / `180.40 ms` Pruned). Controlled back-to-back audit shows Pruned and Original Keras have **virtually identical CPU speed** (`101.54 ms vs 104.89 ms` at `ONEDNN=1`), proving the `180.40 ms` spike and `~2x` TFLite gap were caused by **host CPU load/throttling variance + oneDNN setting mismatch**. Also `mobilenetv2_pruning_report.md:141` claims `"~16x faster"` INT8 latency despite its own table showing **19.9x** (**DISCREPANCY**). |
| **9. Preservation Compliance Check** | **VERIFIED** | Original `.keras`, `.tflite`, INT8 `.tflite`, `evaluation.py`, and `streamlit_app.py` hashes/mtimes remain 100% unaltered. |

---

## 1. Complete Architecture of `models/best_mobilenetv2.keras` (Requirement 1)

**Classification:** **VERIFIED**

Direct deserialization of `models/best_mobilenetv2.keras` (`Keras 3.15.0` loading a `Keras 3.13.2` ZIP archive saved `2026-08-17@08:10:31`) and JSON inspection of its embedded `config.json` establish the complete topology below.

### 1.1 Top-Level Outer Functional Graph (8 Layers)

| Outer Index | Layer Name | Layer Class | Output Shape | Total Params | Trainable Params | Non-Trainable Params | Key Layer Hyperparameters |
|---:|---|---|---|---:|---:|---:|---|
| 0 | `input_layer_1` | `InputLayer` | `(None, 224, 224, 3)` | 0 | 0 | 0 | `batch_shape=(None, 224, 224, 3)` |
| 1 | `mobilenetv2_1.00_224` | `Functional` | `(None, 7, 7, 1280)` | 2,257,984 | 2,223,872 | 34,112 | Nested MobileNetV2 backbone (`154` inner layers; alpha=1.00) |
| 2 | `global_average_pooling2d` | `GlobalAveragePooling2D` | `(None, 1280)` | 0 | 0 | 0 | Spatial average over `7 x 7` feature map |
| 3 | `dense` | `Dense` | `(None, 256)` | 327,936 | 327,936 | 0 | **`units=256`**, `activation='relu'` (`1280*256 + 256`) |
| 4 | `dropout` | `Dropout` | `(None, 256)` | 0 | 0 | 0 | **`rate=0.2`** |
| 5 | `dense_1` | `Dense` | `(None, 64)` | 16,448 | 16,448 | 0 | **`units=64`**, `activation='relu'` (`256*64 + 64`) |
| 6 | `dropout_1` | `Dropout` | `(None, 64)` | 0 | 0 | 0 | **`rate=0.1`** |
| 7 | `dense_2` | `Dense` | `(None, 3)` | 195 | 195 | 0 | **`units=3`**, `activation='softmax'` (`64*3 + 3`) |
| **Total** | — | — | `(None, 3)` | **2,602,563** | **2,568,451** | **34,112** | **100% Verified from runtime + `config.json`** |

**Additional Saved `compile_config` Verified in `best_mobilenetv2.keras`:**
- Optimizer: `Adam` (`learning_rate=2.7e-05`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-07`, `amsgrad=False`)
- Loss: `sparse_categorical_crossentropy` | Metrics: `['accuracy']` | `jit_compile: true`

**Layer-Count Reconciliation Note (`160` vs `8 + 154`):**
- `models/best_mobilenetv2.keras` stores MobileNetV2 as a nested `Functional` sub-model (`154` inner layers) inside an `8`-layer outer graph (`8` outer + `154` inner).
- During fine-tuning, `prune_and_train_mobilenetv2.py` (lines 229–239) built an unnested `tf_keras.Model` directly onto `base_bb.output`, yielding `154 + 6 = 160` flat layers (`2` fewer because the outer `input_layer_1` and wrapper `mobilenetv2_1.00_224` container do not exist in a flat graph).
- Comparing `models/best_mobilenetv2.keras` against `models/best_mobilenetv2_pruned.keras` layer-by-layer confirms **`pruned_structural_match = True`** and **`pruned_total_params = 2,602,563`** (all outer and inner layer names, shapes, Dense dimensions, and Dropout rates match identically).

### 1.2 Complete Nested Backbone Layer Table (`mobilenetv2_1.00_224` — All 154 Layers)

Breakdown of the **154** inner backbone layers (total **2,257,984** parameters = **2,223,872** trainable + **34,112** non-trainable BatchNormalization moving statistics):
- `InputLayer`: 1 | `Conv2D`: 35 (2,125,536 params, all kernels, zero biases) | `BatchNormalization`: 52 (68,224 params: 34,112 trainable gamma/beta + 34,112 non-trainable moving mean/variance) | `ReLU`: 35 | `DepthwiseConv2D`: 17 (64,224 params) | `ZeroPadding2D`: 4 | `Add`: 10.

| # | Layer Name | Layer Type | Output Shape | Total Params | Trainable Params | Non-Trainable Params |
|---:|---|---|---|---:|---:|---:|
| 0 | `input_layer` | `InputLayer` | `(None, 224, 224, 3)` | 0 | 0 | 0 |
| 1 | `Conv1` | `Conv2D` | `(None, 112, 112, 32)` | 864 | 864 | 0 |
| 2 | `bn_Conv1` | `BatchNormalization` | `(None, 112, 112, 32)` | 128 | 64 | 64 |
| 3 | `Conv1_relu` | `ReLU` | `(None, 112, 112, 32)` | 0 | 0 | 0 |
| 4 | `expanded_conv_depthwise` | `DepthwiseConv2D` | `(None, 112, 112, 32)` | 288 | 288 | 0 |
| 5 | `expanded_conv_depthwise_BN` | `BatchNormalization` | `(None, 112, 112, 32)` | 128 | 64 | 64 |
| 6 | `expanded_conv_depthwise_relu` | `ReLU` | `(None, 112, 112, 32)` | 0 | 0 | 0 |
| 7 | `expanded_conv_project` | `Conv2D` | `(None, 112, 112, 16)` | 512 | 512 | 0 |
| 8 | `expanded_conv_project_BN` | `BatchNormalization` | `(None, 112, 112, 16)` | 64 | 32 | 32 |
| 9 | `block_1_expand` | `Conv2D` | `(None, 112, 112, 96)` | 1,536 | 1,536 | 0 |
| 10 | `block_1_expand_BN` | `BatchNormalization` | `(None, 112, 112, 96)` | 384 | 192 | 192 |
| 11 | `block_1_expand_relu` | `ReLU` | `(None, 112, 112, 96)` | 0 | 0 | 0 |
| 12 | `block_1_pad` | `ZeroPadding2D` | `(None, 113, 113, 96)` | 0 | 0 | 0 |
| 13 | `block_1_depthwise` | `DepthwiseConv2D` | `(None, 56, 56, 96)` | 864 | 864 | 0 |
| 14 | `block_1_depthwise_BN` | `BatchNormalization` | `(None, 56, 56, 96)` | 384 | 192 | 192 |
| 15 | `block_1_depthwise_relu` | `ReLU` | `(None, 56, 56, 96)` | 0 | 0 | 0 |
| 16 | `block_1_project` | `Conv2D` | `(None, 56, 56, 24)` | 2,304 | 2,304 | 0 |
| 17 | `block_1_project_BN` | `BatchNormalization` | `(None, 56, 56, 24)` | 96 | 48 | 48 |
| 18 | `block_2_expand` | `Conv2D` | `(None, 56, 56, 144)` | 3,456 | 3,456 | 0 |
| 19 | `block_2_expand_BN` | `BatchNormalization` | `(None, 56, 56, 144)` | 576 | 288 | 288 |
| 20 | `block_2_expand_relu` | `ReLU` | `(None, 56, 56, 144)` | 0 | 0 | 0 |
| 21 | `block_2_depthwise` | `DepthwiseConv2D` | `(None, 56, 56, 144)` | 1,296 | 1,296 | 0 |
| 22 | `block_2_depthwise_BN` | `BatchNormalization` | `(None, 56, 56, 144)` | 576 | 288 | 288 |
| 23 | `block_2_depthwise_relu` | `ReLU` | `(None, 56, 56, 144)` | 0 | 0 | 0 |
| 24 | `block_2_project` | `Conv2D` | `(None, 56, 56, 24)` | 3,456 | 3,456 | 0 |
| 25 | `block_2_project_BN` | `BatchNormalization` | `(None, 56, 56, 24)` | 96 | 48 | 48 |
| 26 | `block_2_add` | `Add` | `(None, 56, 56, 24)` | 0 | 0 | 0 |
| 27 | `block_3_expand` | `Conv2D` | `(None, 56, 56, 144)` | 3,456 | 3,456 | 0 |
| 28 | `block_3_expand_BN` | `BatchNormalization` | `(None, 56, 56, 144)` | 576 | 288 | 288 |
| 29 | `block_3_expand_relu` | `ReLU` | `(None, 56, 56, 144)` | 0 | 0 | 0 |
| 30 | `block_3_pad` | `ZeroPadding2D` | `(None, 57, 57, 144)` | 0 | 0 | 0 |
| 31 | `block_3_depthwise` | `DepthwiseConv2D` | `(None, 28, 28, 144)` | 1,296 | 1,296 | 0 |
| 32 | `block_3_depthwise_BN` | `BatchNormalization` | `(None, 28, 28, 144)` | 576 | 288 | 288 |
| 33 | `block_3_depthwise_relu` | `ReLU` | `(None, 28, 28, 144)` | 0 | 0 | 0 |
| 34 | `block_3_project` | `Conv2D` | `(None, 28, 28, 32)` | 4,608 | 4,608 | 0 |
| 35 | `block_3_project_BN` | `BatchNormalization` | `(None, 28, 28, 32)` | 128 | 64 | 64 |
| 36 | `block_4_expand` | `Conv2D` | `(None, 28, 28, 192)` | 6,144 | 6,144 | 0 |
| 37 | `block_4_expand_BN` | `BatchNormalization` | `(None, 28, 28, 192)` | 768 | 384 | 384 |
| 38 | `block_4_expand_relu` | `ReLU` | `(None, 28, 28, 192)` | 0 | 0 | 0 |
| 39 | `block_4_depthwise` | `DepthwiseConv2D` | `(None, 28, 28, 192)` | 1,728 | 1,728 | 0 |
| 40 | `block_4_depthwise_BN` | `BatchNormalization` | `(None, 28, 28, 192)` | 768 | 384 | 384 |
| 41 | `block_4_depthwise_relu` | `ReLU` | `(None, 28, 28, 192)` | 0 | 0 | 0 |
| 42 | `block_4_project` | `Conv2D` | `(None, 28, 28, 32)` | 6,144 | 6,144 | 0 |
| 43 | `block_4_project_BN` | `BatchNormalization` | `(None, 28, 28, 32)` | 128 | 64 | 64 |
| 44 | `block_4_add` | `Add` | `(None, 28, 28, 32)` | 0 | 0 | 0 |
| 45 | `block_5_expand` | `Conv2D` | `(None, 28, 28, 192)` | 6,144 | 6,144 | 0 |
| 46 | `block_5_expand_BN` | `BatchNormalization` | `(None, 28, 28, 192)` | 768 | 384 | 384 |
| 47 | `block_5_expand_relu` | `ReLU` | `(None, 28, 28, 192)` | 0 | 0 | 0 |
| 48 | `block_5_depthwise` | `DepthwiseConv2D` | `(None, 28, 28, 192)` | 1,728 | 1,728 | 0 |
| 49 | `block_5_depthwise_BN` | `BatchNormalization` | `(None, 28, 28, 192)` | 768 | 384 | 384 |
| 50 | `block_5_depthwise_relu` | `ReLU` | `(None, 28, 28, 192)` | 0 | 0 | 0 |
| 51 | `block_5_project` | `Conv2D` | `(None, 28, 28, 32)` | 6,144 | 6,144 | 0 |
| 52 | `block_5_project_BN` | `BatchNormalization` | `(None, 28, 28, 32)` | 128 | 64 | 64 |
| 53 | `block_5_add` | `Add` | `(None, 28, 28, 32)` | 0 | 0 | 0 |
| 54 | `block_6_expand` | `Conv2D` | `(None, 28, 28, 192)` | 6,144 | 6,144 | 0 |
| 55 | `block_6_expand_BN` | `BatchNormalization` | `(None, 28, 28, 192)` | 768 | 384 | 384 |
| 56 | `block_6_expand_relu` | `ReLU` | `(None, 28, 28, 192)` | 0 | 0 | 0 |
| 57 | `block_6_pad` | `ZeroPadding2D` | `(None, 29, 29, 192)` | 0 | 0 | 0 |
| 58 | `block_6_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 192)` | 1,728 | 1,728 | 0 |
| 59 | `block_6_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 192)` | 768 | 384 | 384 |
| 60 | `block_6_depthwise_relu` | `ReLU` | `(None, 14, 14, 192)` | 0 | 0 | 0 |
| 61 | `block_6_project` | `Conv2D` | `(None, 14, 14, 64)` | 12,288 | 12,288 | 0 |
| 62 | `block_6_project_BN` | `BatchNormalization` | `(None, 14, 14, 64)` | 256 | 128 | 128 |
| 63 | `block_7_expand` | `Conv2D` | `(None, 14, 14, 384)` | 24,576 | 24,576 | 0 |
| 64 | `block_7_expand_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 65 | `block_7_expand_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 66 | `block_7_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 384)` | 3,456 | 3,456 | 0 |
| 67 | `block_7_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 68 | `block_7_depthwise_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 69 | `block_7_project` | `Conv2D` | `(None, 14, 14, 64)` | 24,576 | 24,576 | 0 |
| 70 | `block_7_project_BN` | `BatchNormalization` | `(None, 14, 14, 64)` | 256 | 128 | 128 |
| 71 | `block_7_add` | `Add` | `(None, 14, 14, 64)` | 0 | 0 | 0 |
| 72 | `block_8_expand` | `Conv2D` | `(None, 14, 14, 384)` | 24,576 | 24,576 | 0 |
| 73 | `block_8_expand_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 74 | `block_8_expand_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 75 | `block_8_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 384)` | 3,456 | 3,456 | 0 |
| 76 | `block_8_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 77 | `block_8_depthwise_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 78 | `block_8_project` | `Conv2D` | `(None, 14, 14, 64)` | 24,576 | 24,576 | 0 |
| 79 | `block_8_project_BN` | `BatchNormalization` | `(None, 14, 14, 64)` | 256 | 128 | 128 |
| 80 | `block_8_add` | `Add` | `(None, 14, 14, 64)` | 0 | 0 | 0 |
| 81 | `block_9_expand` | `Conv2D` | `(None, 14, 14, 384)` | 24,576 | 24,576 | 0 |
| 82 | `block_9_expand_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 83 | `block_9_expand_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 84 | `block_9_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 384)` | 3,456 | 3,456 | 0 |
| 85 | `block_9_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 86 | `block_9_depthwise_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 87 | `block_9_project` | `Conv2D` | `(None, 14, 14, 64)` | 24,576 | 24,576 | 0 |
| 88 | `block_9_project_BN` | `BatchNormalization` | `(None, 14, 14, 64)` | 256 | 128 | 128 |
| 89 | `block_9_add` | `Add` | `(None, 14, 14, 64)` | 0 | 0 | 0 |
| 90 | `block_10_expand` | `Conv2D` | `(None, 14, 14, 384)` | 24,576 | 24,576 | 0 |
| 91 | `block_10_expand_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 92 | `block_10_expand_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 93 | `block_10_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 384)` | 3,456 | 3,456 | 0 |
| 94 | `block_10_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 384)` | 1,536 | 768 | 768 |
| 95 | `block_10_depthwise_relu` | `ReLU` | `(None, 14, 14, 384)` | 0 | 0 | 0 |
| 96 | `block_10_project` | `Conv2D` | `(None, 14, 14, 96)` | 36,864 | 36,864 | 0 |
| 97 | `block_10_project_BN` | `BatchNormalization` | `(None, 14, 14, 96)` | 384 | 192 | 192 |
| 98 | `block_11_expand` | `Conv2D` | `(None, 14, 14, 576)` | 55,296 | 55,296 | 0 |
| 99 | `block_11_expand_BN` | `BatchNormalization` | `(None, 14, 14, 576)` | 2,304 | 1,152 | 1,152 |
| 100 | `block_11_expand_relu` | `ReLU` | `(None, 14, 14, 576)` | 0 | 0 | 0 |
| 101 | `block_11_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 576)` | 5,184 | 5,184 | 0 |
| 102 | `block_11_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 576)` | 2,304 | 1,152 | 1,152 |
| 103 | `block_11_depthwise_relu` | `ReLU` | `(None, 14, 14, 576)` | 0 | 0 | 0 |
| 104 | `block_11_project` | `Conv2D` | `(None, 14, 14, 96)` | 55,296 | 55,296 | 0 |
| 105 | `block_11_project_BN` | `BatchNormalization` | `(None, 14, 14, 96)` | 384 | 192 | 192 |
| 106 | `block_11_add` | `Add` | `(None, 14, 14, 96)` | 0 | 0 | 0 |
| 107 | `block_12_expand` | `Conv2D` | `(None, 14, 14, 576)` | 55,296 | 55,296 | 0 |
| 108 | `block_12_expand_BN` | `BatchNormalization` | `(None, 14, 14, 576)` | 2,304 | 1,152 | 1,152 |
| 109 | `block_12_expand_relu` | `ReLU` | `(None, 14, 14, 576)` | 0 | 0 | 0 |
| 110 | `block_12_depthwise` | `DepthwiseConv2D` | `(None, 14, 14, 576)` | 5,184 | 5,184 | 0 |
| 111 | `block_12_depthwise_BN` | `BatchNormalization` | `(None, 14, 14, 576)` | 2,304 | 1,152 | 1,152 |
| 112 | `block_12_depthwise_relu` | `ReLU` | `(None, 14, 14, 576)` | 0 | 0 | 0 |
| 113 | `block_12_project` | `Conv2D` | `(None, 14, 14, 96)` | 55,296 | 55,296 | 0 |
| 114 | `block_12_project_BN` | `BatchNormalization` | `(None, 14, 14, 96)` | 384 | 192 | 192 |
| 115 | `block_12_add` | `Add` | `(None, 14, 14, 96)` | 0 | 0 | 0 |
| 116 | `block_13_expand` | `Conv2D` | `(None, 14, 14, 576)` | 55,296 | 55,296 | 0 |
| 117 | `block_13_expand_BN` | `BatchNormalization` | `(None, 14, 14, 576)` | 2,304 | 1,152 | 1,152 |
| 118 | `block_13_expand_relu` | `ReLU` | `(None, 14, 14, 576)` | 0 | 0 | 0 |
| 119 | `block_13_pad` | `ZeroPadding2D` | `(None, 15, 15, 576)` | 0 | 0 | 0 |
| 120 | `block_13_depthwise` | `DepthwiseConv2D` | `(None, 7, 7, 576)` | 5,184 | 5,184 | 0 |
| 121 | `block_13_depthwise_BN` | `BatchNormalization` | `(None, 7, 7, 576)` | 2,304 | 1,152 | 1,152 |
| 122 | `block_13_depthwise_relu` | `ReLU` | `(None, 7, 7, 576)` | 0 | 0 | 0 |
| 123 | `block_13_project` | `Conv2D` | `(None, 7, 7, 160)` | 92,160 | 92,160 | 0 |
| 124 | `block_13_project_BN` | `BatchNormalization` | `(None, 7, 7, 160)` | 640 | 320 | 320 |
| 125 | `block_14_expand` | `Conv2D` | `(None, 7, 7, 960)` | 153,600 | 153,600 | 0 |
| 126 | `block_14_expand_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 127 | `block_14_expand_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 128 | `block_14_depthwise` | `DepthwiseConv2D` | `(None, 7, 7, 960)` | 8,640 | 8,640 | 0 |
| 129 | `block_14_depthwise_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 130 | `block_14_depthwise_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 131 | `block_14_project` | `Conv2D` | `(None, 7, 7, 160)` | 153,600 | 153,600 | 0 |
| 132 | `block_14_project_BN` | `BatchNormalization` | `(None, 7, 7, 160)` | 640 | 320 | 320 |
| 133 | `block_14_add` | `Add` | `(None, 7, 7, 160)` | 0 | 0 | 0 |
| 134 | `block_15_expand` | `Conv2D` | `(None, 7, 7, 960)` | 153,600 | 153,600 | 0 |
| 135 | `block_15_expand_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 136 | `block_15_expand_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 137 | `block_15_depthwise` | `DepthwiseConv2D` | `(None, 7, 7, 960)` | 8,640 | 8,640 | 0 |
| 138 | `block_15_depthwise_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 139 | `block_15_depthwise_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 140 | `block_15_project` | `Conv2D` | `(None, 7, 7, 160)` | 153,600 | 153,600 | 0 |
| 141 | `block_15_project_BN` | `BatchNormalization` | `(None, 7, 7, 160)` | 640 | 320 | 320 |
| 142 | `block_15_add` | `Add` | `(None, 7, 7, 160)` | 0 | 0 | 0 |
| 143 | `block_16_expand` | `Conv2D` | `(None, 7, 7, 960)` | 153,600 | 153,600 | 0 |
| 144 | `block_16_expand_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 145 | `block_16_expand_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 146 | `block_16_depthwise` | `DepthwiseConv2D` | `(None, 7, 7, 960)` | 8,640 | 8,640 | 0 |
| 147 | `block_16_depthwise_BN` | `BatchNormalization` | `(None, 7, 7, 960)` | 3,840 | 1,920 | 1,920 |
| 148 | `block_16_depthwise_relu` | `ReLU` | `(None, 7, 7, 960)` | 0 | 0 | 0 |
| 149 | `block_16_project` | `Conv2D` | `(None, 7, 7, 320)` | 307,200 | 307,200 | 0 |
| 150 | `block_16_project_BN` | `BatchNormalization` | `(None, 7, 7, 320)` | 1,280 | 640 | 640 |
| 151 | `Conv_1` | `Conv2D` | `(None, 7, 7, 1280)` | 409,600 | 409,600 | 0 |
| 152 | `Conv_1_bn` | `BatchNormalization` | `(None, 7, 7, 1280)` | 5,120 | 2,560 | 2,560 |
| 153 | `out_relu` | `ReLU` | `(None, 7, 7, 1280)` | 0 | 0 | 0 |

---

## 2. Resolution of `Dense(512) -> Dense(128)` vs `Dense(256) -> Dense(64)` Discrepancy (Requirement 2)

**Classification:** **DISCREPANCY** identified in `dermavision_original_model_verification_report.md` (Section 2A) and downstream user prompts; **VERIFIED** in `prune_and_train_mobilenetv2.py`, `evaluate_pruned_mobilenetv2.py`, and `results/pruning/mobilenetv2_pruning_report.md`.

### Evidence & Forensic Root-Cause Analysis

Four independent facts definitively resolve the contradiction:

1. **Direct Weight & `config.json` Inspection (`VERIFIED`):**
   Both live layer attributes and the raw serialized JSON inside `models/best_mobilenetv2.keras` prove the classification head is:
   `GlobalAveragePooling2D (1280) -> Dense(256, relu) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.1) -> Dense(3, softmax)`.
2. **Mathematical Parameter Conservation (`VERIFIED`):**
   - **Actual (`256 -> 64 -> 3`):**
     - Backbone: `2,257,984`
     - `Dense(256)`: `1,280 x 256 + 256 = 327,936`
     - `Dense(64)`: `256 x 64 + 64 = 16,448`
     - `Dense(3)`: `64 x 3 + 3 = 195`
     - **Sum:** `2,257,984 + 344,579 = 2,602,563` -> **EXACT match** to `orig_model.count_params()`.
   - **Hypothetical (`512 -> 128 -> 3`):**
     - `Dense(512)`: `1,280 x 512 + 512 = 655,872`
     - `Dense(128)`: `512 x 128 + 128 = 65,664`
     - `Dense(3)`: `128 x 3 + 3 = 387`
     - **Hypothetical Sum:** `2,257,984 + 721,923 = 2,979,907` (`+377,344` params, impossible inside `best_mobilenetv2.keras`).
3. **Internal Self-Contradiction in `dermavision_original_model_verification_report.md` (`DISCREPANCY`):**
   Section 2A (lines 29–33) wrote `Dense(512) -> Dropout(0.3) -> Dense(128) -> Dropout(0.2) -> Dense(3)`, even though Section 6 (line 138) of the *exact same report* recorded `Total Parameters: 2,602,563` (`2,568,451` trainable + `34,112` non-trainable). Those numbers cannot coexist.
4. **Log Trace Showing How the Error Originated (`VERIFIED`):**
   Inspection of the pre-deployment audit execution logs explains the error:
   - `task-868.log` (lines 7–11) crashed with `AttributeError: 'InputLayer' object has no attribute 'output_shape'` before printing head layer shapes.
   - `task-877.log` (lines 5–14) avoided shapes and printed only `layer.name`, class name, and activation object (`dense Dense <function relu>`, `dropout Dropout None`, `dense_1 Dense <function relu>`, `dropout_1 Dropout None`, `dense_2 Dense <function softmax>`) without printing `.units` or `.rate`.
   - The `Dense(512) -> Dropout(0.3) -> Dense(128) -> Dropout(0.2)` head (used separately in `build_kaggle_notebook.py` lines 224–228 for Revision V1.1 staged training and echoed in later prompt text) was erroneously assumed in prose without checking `.units`, whereas `train_mobilenetv2_revision_v1.py`, `results/revision_v1_training_technical_review.txt`, `prune_and_train_mobilenetv2.py`, and `results/pruning/mobilenetv2_pruning_report.md` all accurately document `Dense(256) -> Dropout(0.2) -> Dense(64) -> Dropout(0.1) -> Dense(3)`.

---

## 3. Initialization & Checkpoint Lineage Verification (Requirement 3)

### 3.1 Summary of Lineage Evidence

- **Script Weight-Loading Logic — `VERIFIED`:**  
  `prune_and_train_mobilenetv2.py` (lines 225–254) explicitly loads `models/best_mobilenetv2.keras` via `keras.models.load_model`, copies weights layer-by-layer for all `153` non-input backbone layers and all `3` head `Dense` layers into an equivalent `tf_keras` model, verifies functional output parity (`max_parity_diff = 6.52e-08`, threshold `< 1e-5`), and selectively wraps only `tf_keras.layers.Conv2D` (35) and `tf_keras.layers.Dense` (3) with `tfmot.sparsity.keras.prune_low_magnitude`. The `17` `DepthwiseConv2D` layers and `52` `BatchNormalization` layers remain unwrapped (`38` wrapped, `122` preserved in the `160`-layer flat model).
- **Epoch 5 Stripping Parity (`max_diff = 0.0`) — `VERIFIED`:**  
  Independently loading `results/pruning/checkpoints/mobilenetv2_pruned_epoch_05.h5` (`41,810,616 B`, `mtime 2026-11-10 09:22:02`) inside `tfmot.sparsity.keras.prune_scope()`, applying `tfmot.sparsity.keras.strip_pruning()`, and comparing every single weight array across all `153` weighted backbone layers + `3` dense head layers against `models/best_mobilenetv2_pruned.keras` (`31,673,383 B`, `mtime 2026-11-10 09:22:07`) yields **`max_abs_diff = 0.00e+00`**. Thus `best_mobilenetv2_pruned.keras` is bit-for-bit the stripped Epoch 5 checkpoint.
- **Independent HDF5 Epoch 1–5 Sparsity Progression — `VERIFIED`:**  
  Direct `h5py` inspection of the `38` wrapped kernels inside `results/pruning/checkpoints/mobilenetv2_pruned_epoch_01..05.h5` confirms exact agreement with `results/pruning/mobilenetv2_pruning_history.csv`:

| Epoch Checkpoint | File Mtime | SHA-256 (First 16 hex) | Conv2D Zeros (/ 2,125,536) | Dense Zeros (/ 344,256) | Total Prunable Zeros (/ 2,469,792) | Measured Kernel Sparsity | Logged Val Accuracy |
|:---:|---|---|---:|---:|---:|---:|---:|
| `epoch_01.h5` | `2026-11-10 08:48:59` | `e46866d72f5c8f75...` | 502,776 | 81,432 | 584,208 | **23.6541%** | **90.60%** |
| `epoch_02.h5` | `2026-11-10 08:57:21` | `578ec698af64e433...` | 815,091 | 132,015 | 947,106 | **38.3476%** | 84.61% |
| `epoch_03.h5` | `2026-11-10 09:08:08` | `3d3e3f1aa8ed3f52...` | 982,280 | 159,094 | 1,141,374 | **46.2134%** | 68.38% |
| `epoch_04.h5` | `2026-11-10 09:14:55` | `0059abfc89eba8a7...` | 1,049,705 | 170,012 | 1,219,717 | **49.3854%** | 81.48% |
| `epoch_05.h5` | `2026-11-10 09:22:02` | `35d23335cf8f92d0...` | 1,062,706 | 172,119 | 1,234,825 | **49.9971%** | **83.49%** |

- **Mathematical Non-Zero Weight Similarity Proof — `VERIFIED`:**  
  Comparing the *surviving (non-zero)* kernel parameters of `models/best_mobilenetv2_pruned.keras` against the exact corresponding elements of `models/best_mobilenetv2.keras` proves continuous low-learning-rate (`1e-5`) fine-tuning rather than training from scratch or ImageNet re-initialization:
  - Across all **52** backbone `Conv2D` & `DepthwiseConv2D` kernels: mean cosine similarity = **`0.999989`**, minimum cosine similarity = **`0.999934`**.
  - Head `dense` non-zero weights: cosine similarity = **`0.999975`** (`max |delta| = 2.69e-03`, `mean |delta| = 2.07e-04`).
  - Head `dense_1` non-zero weights: cosine similarity = **`0.999992`** (`max |delta| = 3.85e-03`, `mean |delta| = 2.18e-04`).
  - Head `dense_2` non-zero weights: cosine similarity = **`0.999995`** (`max |delta| = 2.88e-03`, `mean |delta| = 5.38e-04`).

### 3.2 Caveats, Metadata Gaps & Checkpoint Selection Discrepancies

- **No Embedded Lineage Metadata Inside `.keras` Archive — `UNVERIFIED` (from standalone `.keras` alone):**  
  Inspecting `models/best_mobilenetv2_pruned.keras` (`ZIP` -> `metadata.json`) shows only `{"keras_version": "3.15.0", "date_saved": "2026-11-10@09:22:07"}`. Neither `best_mobilenetv2.keras` (`keras 3.13.2`, `date_saved 2026-08-17@08:10:31`) nor `best_mobilenetv2_pruned.keras` stores parent checkpoint hashes or lineage provenance fields. Provenance is therefore proven by code, intermediate `.h5` files, and mathematical weight similarity, *not* by embedded archive metadata. Similarly, no written pre-run SHA-256 log was recorded before pruning started (**`UNVERIFIED`** via prior hash log, though untouched OS `mtime = 2026-08-17 13:57:56` and Keras `3.13.2` internal timestamp corroborate zero mutation).
- **Checkpoint Selection Policy (`>=49.0%` Sparsity Filter vs Global Max Val Accuracy) — `DISCREPANCY` / Nuance:**  
  - Independent evaluation of the stripped **Epoch 1** checkpoint (`mobilenetv2_pruned_epoch_01.h5`) confirms **`90.5986%` validation accuracy** (`2,255 / 2,489`, Melanoma recall `88.05%`), which is `+7.11 pp` higher than Epoch 5 (`83.49%`), but Epoch 1 only has **`23.6541%`** prunable sparsity.
  - `prune_and_train_mobilenetv2.py` (line 321) intentionally filtered candidates to `prunable_sparsity_percent >= 49.0` before taking `max(val_accuracy)`, selecting **Epoch 5 (`83.49%` @ `49.9971%` sparsity)**. While this deviation from naive "best val_accuracy across all epochs" is transparently explained in `results/pruning/mobilenetv2_pruning_report.md` (Section 3, lines 57–61), note a minor internal docstring discrepancy inside `prune_and_train_mobilenetv2.py`: line 24 states `sparsity >= 49.5%`, whereas line 321 implements `>= 49.0%` (impact is nil because Epoch 4 is `49.3854%` with lower accuracy `81.48%` and Epoch 5 is `49.9971%` with `83.49%`).

---

## 4. Pipeline Parity: Class Mapping, Preprocessing, Paths & Labels (Requirement 4)

**Classification:** **VERIFIED** (with historical Kaggle Ep-50 log distinction noted)

| Pipeline Attribute | Verified Specification | Verification Evidence Across Code & Artifacts | Status |
|---|---|---|---|
| **Class-to-Index Mapping** | `0: BCC`, `1: Melanoma`, `2: Nevus` | Identical in `models/class_names.json` (`{"0": "BCC", "1": "Melanoma", "2": "Nevus"}`), `evaluation.py:42-43`, `quantize_and_evaluate_int8.py:45-46`, `prune_and_train_mobilenetv2.py:18`, and `evaluate_pruned_mobilenetv2.py:57-58`. | **VERIFIED** |
| **Input Dimensions & Dtype** | `(None, 224, 224, 3)`, `float32` | Exact match in `best_mobilenetv2.keras`, `best_mobilenetv2_pruned.keras`, and `IMG_SIZE = (224, 224)` across all evaluation pipelines. | **VERIFIED** |
| **Image Preprocessing** | RGB JPEG decode -> `tf.image.resize((224,224))` -> `cast(float32) / 255.0` | Identical tensor code in `evaluation.py:109-114`, `quantize_and_evaluate_int8.py:169-174`, `prune_and_train_mobilenetv2.py:195-200`, `evaluate_pruned_mobilenetv2.py:246-251`, and `preprocessing.py:24-28` (`PIL` resize + `/255.0`). `mobilenet_v2.preprocess_input` (`[-1, 1]`) is NOT used in either model. | **VERIFIED** |
| **Partition Counts & Balance** | **9,956** train / **2,489** val (`BCC=665`, `Mel=904`, `Nev=920`) | Reconstructed deterministically by `prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", img_dir)` (`NV` capped at `4600`, stratified 80:20 split, `random_state=42`). | **VERIFIED** |
| **Path Resolution & Label Parity** | Exact 1-to-1 sample order & label match | Independent check comparing `img_dir="ISIC_2019_Training_Input"` (used in `evaluation.py`) vs `"ISIC_2019_Training_Input/ISIC_2019_Training_Input"` (used in INT8 & pruning scripts) confirmed normalized-path identity (`True`) and label identity (`True`). | **VERIFIED** |
| **Output Layer & Prediction Rule** | `Dense(3, softmax)` -> `argmax(axis=1)` | Softmax row sums verified empirically across all `2,489` samples: Original `[0.99999982, 1.00000012]`, Pruned `[0.99999982, 1.00000012]`. | **VERIFIED** |

*Historical Context Note:* As already documented in `dermavision_original_model_verification_report.md` Section 3A, the Kaggle training CSV log (`results/mobilenetv2_epoch_history.csv`) ended at Epoch 50 with `90.12%` validation accuracy (and peak `90.16%` at Ep 45–46), whereas deterministic post-hoc evaluation of `models/best_mobilenetv2.keras` via `evaluation.py` yields **`90.96%` (`2,264 / 2,489`)**, verified identically across `metrics/mobilenet_val_probabilities.npy`, `metrics/final_metrics.json`, the INT8 report, the pruning report, and this audit.

---

## 5 & 6. Independent Evaluation: Accuracy, Per-Class Metrics, Confusion Matrices & Sparsity (Requirements 5 & 6)

**Classification:** **VERIFIED** (100% numeric agreement across every metric and cell)

Independent evaluation was executed in a clean subprocess with `TF_ENABLE_ONEDNN_OPTS="0"` and `batch_size=64` over the full **2,489** validation images (`val_imgs`), comparing directly against `results/pruning/mobilenetv2_pruning_report.md` and `results/pruning/mobilenetv2_pruning_metrics.json` (which ran with `TF_ENABLE_ONEDNN_OPTS="1"`).

### 5.1 Overall Classification Metrics & Stripped Epoch 1 Reference

| Model Evaluated | Correct / Total | Independently Verified Accuracy | Reported Accuracy | Macro Precision | Macro Recall | Macro F1 | Verification Status |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Original FP32 (`best_mobilenetv2.keras`)** | **2,264 / 2,489** | **90.9602%** | 90.96% | 91.11% (`91.1092%`) | 91.45% (`91.4532%`) | 91.27% (`91.2665%`) | **VERIFIED** (Exact) |
| **Pruned Epoch 5 (`best_mobilenetv2_pruned.keras`)** | **2,078 / 2,489** | **83.4873%** | 83.49% | 84.84% (`84.8398%`) | 83.33% (`83.3293%`) | 83.55% (`83.5474%`) | **VERIFIED** (Exact) |
| **Delta (Pruned Ep5 vs Original FP32)** | **-186 samples** | **-7.4729 pp** | -7.47 pp | -6.27 pp | -8.12 pp | -7.72 pp | **VERIFIED** (Exact) |
| **Reference: Epoch 1 Stripped (`epoch_01.h5`, 23.65% sparsity)** | 2,255 / 2,489 | **90.5986%** | 90.60% (CSV) | 90.96% (`90.9616%`) | 90.91% (`90.9106%`) | 90.93% (`90.9349%`) | **VERIFIED** |

### 5.2 Per-Class Verification (Precision, Recall / Sensitivity, F1-Score)

| Class (Support) | Metric | Original FP32 (Verified / Reported) | Pruned Ep5 (Verified / Reported) | Delta (Pruned - Orig) | Epoch 1 Reference | Status |
|---|---|---:|---:|---:|---:|:---:|
| **BCC (0)** (`n=665`) | Precision | **93.0435%** / 93.04% | **89.2508%** / 89.25% | -3.79 pp | 94.56% | **VERIFIED** |
| | Recall | **96.5414%** / 96.54% (`642/665`) | **82.4060%** / 82.41% (`548/665`) | -14.14 pp (`-94`) | 94.14% | **VERIFIED** |
| | F1-Score | **94.7601%** / 94.76% | **85.6919%** / 85.69% | -9.07 pp | 94.35% | **VERIFIED** |
| **Melanoma (1)** (`n=904`) | Precision | **89.0271%** / 89.03% | **87.9470%** / 87.95% | -1.08 pp | 87.19% | **VERIFIED** |
| | **Recall (Sensitivity)** | **87.0575%** / 87.06% (`787/904`) | **73.4513%** / 73.45% (`664/904`) | **-13.61 pp** (`-123`) | **88.05%** (`796/904`) | **VERIFIED** |
| | F1-Score | **88.0313%** / 88.03% | **80.0482%** / 80.05% | -7.98 pp | 87.62% | **VERIFIED** |
| **Nevus (2)** (`n=920`) | Precision | **91.2568%** / 91.26% | **77.3214%** / 77.32% | **-13.94 pp** | 91.14% | **VERIFIED** |
| | Recall | **90.7609%** / 90.76% (`835/920`) | **94.1304%** / 94.13% (`866/920`) | +3.37 pp (`+31`) | 90.54% | **VERIFIED** |
| | F1-Score | **91.0082%** / 91.01% | **84.9020%** / 84.90% | -6.11 pp | 90.84% | **VERIFIED** |

### 5.3 Regenerated Confusion Matrices (Independently Verified)

**A. Original FP32 (`models/best_mobilenetv2.keras`) — Verified Exact Match:**
```text
                Predicted BCC    Predicted Melanoma    Predicted Nevus    Total
True BCC             642              21                    2                665
True Melanoma         39             787                   78                904
True Nevus             9              76                  835                920
```

**B. Pruned Epoch 5 (`models/best_mobilenetv2_pruned.keras`) — Verified Exact Match:**
```text
                Predicted BCC    Predicted Melanoma    Predicted Nevus    Total
True BCC             548              39                   78                665
True Melanoma         64             664                  176                904
True Nevus             2              52                  866                920
```

**C. Stripped Epoch 1 Reference (`mobilenetv2_pruned_epoch_01.h5`, 23.65% sparsity):**
```text
                Predicted BCC    Predicted Melanoma    Predicted Nevus    Total
True BCC             626              37                    2                665
True Melanoma         29             796                   79                904
True Nevus             7              80                  833                920
```

**Clinical Risk Diagnostic (`VERIFIED`):** At ~50% sparsity over only 5 fine-tuning epochs (`Adam 1e-5` without constant-sparsity recovery epochs), predictions shifted substantially toward the `Nevus` class: Melanoma false-negatives misclassified as benign `Nevus` more than doubled (`78 -> 176`), and `BCC` misclassified as `Nevus` jumped (`2 -> 78`), driving `Nevus` precision down from `91.26%` to `77.32%` while cutting Melanoma sensitivity from `87.06%` to `73.45%`.

### 5.4 Independent Element-Wise Sparsity & Gzip-9 Size Verification

| Tensor Group | Audited Zeros | Total Elements | Audited Sparsity | Reported Sparsity | Status |
|---|---:|---:|---:|---:|:---:|
| `Conv2D` Kernels (35 layers) | 1,062,706 | 2,125,536 | **49.9971%** (`49.997083%`) | 49.9971% | **VERIFIED** |
| `Dense` Kernels (3 layers) | 172,119 | 344,256 | **49.9974%** (`49.997386%`) | 49.9974% | **VERIFIED** |
| **Total Eligible Prunable** | **1,234,825** | **2,469,792** | **49.9971%** (`49.997125%`) | **49.9971%** | **VERIFIED** |
| Unpruned (`DepthwiseConv2D`, `BN`, biases) | 0 | 132,771 | 0.0000% | 0.0000% | **VERIFIED** |
| **Complete Model Footprint** | **1,234,825** | **2,602,563** | **47.4465%** (`47.446498%`) | **47.4465%** | **VERIFIED** |

Independent `gzip.compress(..., compresslevel=9)` byte counts also matched `mobilenetv2_pruning_metrics.json` 100%: Original FP32 **`28,229,451 B`**, Dynamic-Range TFLite **`2,501,547 B`**, Full INT8 TFLite **`2,553,496 B`**, Pruned Keras **`6,009,286 B`** (**78.71%** smaller archive than gzipped FP32 despite equal ~`30.21 MB` raw dense array ZIP size).

---

## 7 & 8. Latency Benchmark Methodology & Cross-Report Comparison (Requirements 7 & 8)

### 7.1 Benchmark Methodology & Hardware Specification Audit

| Methodology Aspect | `quantize_and_evaluate_int8.py` (INT8 Report) | `evaluate_pruned_mobilenetv2.py` (Pruning Report) | Status |
|---|---|---|---|
| **Execution Timestamp** | `2026-11-10 08:18:17` (`task-957.log`) | `2026-11-10 09:29:42` (`task-1271.log`) | **VERIFIED** |
| **OS & CPU Architecture** | Windows 11 (`10.0.26300`, AMD64); `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel` (`10` physical / `12` logical cores, `15.65 GB` RAM; ISA: `SSE3/SSE4.1/SSE4.2/AVX/AVX2/AVX_VNNI/FMA`); CPU-only | Same host machine & CPU-only execution (`TF 2.21.0`, TFLite XNNPACK delegate enabled) | **VERIFIED** (Exact Intel SKU branding `UNVERIFIED`, CIM blocked) |
| **Environment Variable (`TF_ENABLE_ONEDNN_OPTS`)** | **`"0"`** (oneDNN disabled — line 32) | **`"1"`** (oneDNN enabled — line 44) | **MAJOR DISCREPANCY** |
| **Warm-up & Timed Iterations** | `50` warm-up + `200` timed sequential calls (`batch_size=1`) over `val_images_fp32[:200]` | Identical (`50` warm-up + `200` timed sequential calls over `val_images_fp32[:200]`) | **VERIFIED** |
| **Timer API & Statistics** | `time.perf_counter()`, `np.mean` & population `np.std` (`ddof=0`) | Identical | **VERIFIED** |
| **Model Loading Included?** | **NO** (loaded before timed loop) | **NO** | **VERIFIED** |
| **Image Preprocessing Included?** | **NO** for JPEG decode, resize `(224,224)`, `/255.0`. **YES** for INT8 numpy quantize/dequantize arithmetic | Identical: decode/resize/`/255.0` **excluded**; INT8 quantize/dequantize **included** inside loop | **VERIFIED** |
| **Keras Call Pattern & XLA** | Eager `model(inp, training=False)` (`jit_compile=True` from `best_mobilenetv2.keras` compile config) | Identical | **VERIFIED** |

### 7.2 Cross-Report Comparison & Controlled Back-to-Back Re-Benchmarks

To resolve why all latencies shifted dramatically between `08:18:17` (INT8 report) and `09:29:42` (Pruning report), and whether Pruned Keras is genuinely slower than FP32 Keras (`180.40 ms` vs `131.77 ms`), we ran back-to-back controlled benchmarks (`50` warm-up + `200` timed, `TF` default threading `intra_op=0, inter_op=0`) on the exact same machine under both `TF_ENABLE_ONEDNN_OPTS=0` and `TF_ENABLE_ONEDNN_OPTS=1`:

| Benchmark Source | `ONEDNN` Setting | Measured CPU Load | Original FP32 Keras (ms) | Dynamic-Range TFLite (ms) | Full INT8 TFLite (ms) | Pruned FP32 Keras (ms) | FP32 / INT8 Ratio |
|---|:---:|---:|---:|---:|---:|---:|---:|
| **INT8 Report (`08:18:17`)** | `"0"` | *Unlogged* (`UNVERIFIED`) | **269.56 ± 83.04** | **73.74 ± 9.22** | **16.35 ± 2.43** | *N/A* | **16.5×** (Reported: 16.5×) |
| **Pruning Report (`09:29:42`)** | `"1"` | *Unlogged* (`UNVERIFIED`) | **131.77 ± 17.16** | **38.17 ± 10.91** | **6.63 ± 2.29** | **180.40 ± 34.64** | **19.9×** (*Text says ~16×*) |
| **Independent Audit Run A (`09:55`)** | `"0"` | 37.8% (`1300 MHz` idle read) | **160.07 ± 25.10** | **33.49 ± 4.89** | **6.85 ± 1.56** | **154.46 ± 22.60** | **23.4×** |
| **Independent Audit Run B (`09:57`)** | `"1"` | 11.6% (`1300 MHz` idle read) | **104.89 ± 9.95** | **27.76 ± 1.77** | **4.51 ± 0.20** | **101.54 ± 3.44** | **23.3×** |

### 7.3 All Identified Differences & Discrepancies Between Reports

1. **`TF_ENABLE_ONEDNN_OPTS` Configuration Mismatch (`MAJOR DISCREPANCY`):**
   `quantize_and_evaluate_int8.py:32` explicitly disabled oneDNN (`"0"`), whereas `evaluate_pruned_mobilenetv2.py:44` explicitly enabled oneDNN (`"1"`). Controlled Audit Runs A vs B confirm enabling oneDNN materially accelerates Keras FP32 CPU inference (**`160.07 ms -> 104.89 ms`**, `-34.5%`).
2. **CPU Contention / Thermal / Hybrid P-Core vs E-Core Scheduling Variance (`MAJOR DISCREPANCY`):**
   Notice that TFLite (which runs on XNNPACK rather than oneDNN) was *also* **`~2.5×` slower** at `08:18` than at `09:29` (`16.35 ms vs 6.63 ms` INT8; `73.74 ms vs 38.17 ms` Dynamic-Range), and FP32 Keras had a massive standard deviation at `08:18` (`±83.04 ms`, over 30% CV, executed right after TFLite conversion and 4,978 sequential TFLite invocations inside the same process). On an Intel 12th-Gen Alder Lake hybrid CPU running Windows 11, transient background CPU load, thermal/power state, and Windows scheduler placement across Performance vs Efficiency cores strongly impact single-threaded/batch=1 runtimes. *(Exact background processes, power plan, and thermal state at 08:18 and 09:29 remain **`UNVERIFIED`** post hoc because neither script logged CPU load or power/clock telemetry.)*
3. **Apparent "+37% Slowdown" of Pruned Keras (`180.40 ms` vs `131.77 ms`) Was Run Noise (`DISCREPANCY` in Raw Measurement; Core Architectural Finding `VERIFIED`):**
   When benchmarked back-to-back in our controlled runs, Pruned Keras and Original FP32 Keras execute at virtually **identical latency**: **`154.46 ms vs 160.07 ms`** (`ONEDNN=0`) and **`101.54 ms vs 104.89 ms`** (`ONEDNN=1`). Standard TensorFlow CPU dense kernels still execute the full dense matrix multiplications regardless of 50% zeros, so the pruning report's headline architectural conclusion (**unstructured pruning provides zero CPU inference speedup without a sparse runtime**) is **VERIFIED**, even though the `180.40 ± 34.64 ms` single-run spike over `131.77 ± 17.16 ms` was noisy variance rather than structural slowdown.
4. **In-Report Speedup Text Mismatch (`DISCREPANCY`):**
   `results/pruning/mobilenetv2_pruning_report.md` line 141 states Full INT8 achieves **`"~16x faster"`** CPU latency—a figure carried over from the INT8 report (`269.56 / 16.35 = 16.49×`), whereas in that very report's Table 7 (`131.77 / 6.63`), the actual same-run speedup ratio was **`19.9×`** (and **`23.3×`** in our low-load controlled `ONEDNN=1` run).
5. **Minor Code Hygiene & Post-Run Formatting Differences (`VERIFIED`):**
   - `evaluate_pruned_mobilenetv2.py` additionally evaluates `gzip -9` sizes across **4** models (`28,229,451 B` down to `6,009,286 B`), whereas the INT8 report reported only raw disk sizes across **3** models.
   - During the `09:29:42` run (`task-1271.log`), `evaluate_pruned_mobilenetv2.py` triggered `SyntaxWarning: invalid escape sequence '\p'` (ending at line 569) from `\pm` literals; the script (`mtime 09:29:59`) currently uses Unicode `±` (lines 541–544), while `mobilenetv2_pruning_report.md` retains `$\pm$` (lines 133–136).
   - `quantize_and_evaluate_int8.py:443-448` (`mtime 08:10:57`) contains duplicate template rows for Melanoma and Nevus that do not appear in the saved `results/int8/mobilenetv2_int8_verification_report.md` (`mtime 08:18:58`, 41s after `08:18:17` report header timestamp).

---

## 9. Strict Preservation Compliance Check

**Classification:** **VERIFIED**

All pre-existing model checkpoints, evaluation scripts, reports, and frontend code were verified unchanged via SHA-256 checksums before and after audit execution:

| Artifact Path | Role | Size (Bytes) | Verified SHA-256 Digest | Status |
|---|---|---:|---|:---:|
| `models/best_mobilenetv2.keras` | Original FP32 Checkpoint | 31,673,490 | `ebbd575e3d7b340d8d806515abc918cc5260b8cd4b8ef4c41c93f9582ebf33d4` | **VERIFIED Untouched** (`mtime 2026-08-17 13:57:56`) |
| `models/best_mobilenetv2.tflite` | Original Dynamic-Range TFLite | 2,877,224 | `acf368659274be0fca0bfd6d4715f9a515e7d701d44592aee33fae4e73b2b0eb` | **VERIFIED Untouched** |
| `models/best_mobilenetv2_full_int8.tflite` | Full INT8 TFLite | 3,083,384 | `00924d9d127fd73438ba526ac6bbe28ac3fdc12ee3fff4272212aa82ae075da7` | **VERIFIED Untouched** |
| `models/best_mobilenetv2_pruned.keras` | Pruned Ep5 Stripped Keras | 31,673,383 | `0364f6c4d152c7ccf6d555aadd49eb82bf293f52983bb29bbc73eee4acc3814f` | **VERIFIED Untouched** (`mtime 2026-11-10 09:22:07`) |
| `models/class_names.json` | Canonical Class Mapping | 57 | `8a1823d38491c8ef2f0ee2b093bf200163ddc286406fb3dc62cb220539ffbdcd` | **VERIFIED Untouched** |
| `evaluation.py` | Baseline Evaluation Script | 20,755 | `55b7c934ebad767c0208e77c5300a314d78ab0e68883846acfeae0e813a8a78b` | **VERIFIED Untouched** |
| `streamlit_app.py` | Frontend Application | 91,483 | `925f7505bd7b726d94b0763589f9e22aeaca039f0ead7dae2806507e16bbe08d` | **VERIFIED Untouched** |
| `prune_and_train_mobilenetv2.py` | Pruning Training Pipeline | 19,708 | `467ad0c23da8e7ec82c577540922757f15f7b2dc4557d3b5a86afa7893166a6a` | **VERIFIED Untouched** |
| `evaluate_pruned_mobilenetv2.py` | Pruning Eval Script | 35,747 | `3d2da635dfe6fde6afd9afea36903021bee67f848e8941b71b441c14c7438452` | **VERIFIED Untouched** |
| `quantize_and_evaluate_int8.py` | INT8 Quant & Eval Script | 30,055 | `f6663e9c41d7f3930a0563ca700d79f806d25466bade89ce0982d55a65b80f99` | **VERIFIED Untouched** |
| `results/pruning/mobilenetv2_pruning_report.md` | Original Pruning Report | 12,051 | `40a9790892251047608a02c334e3373458ab97e5d5f4c90877309edaeed9b6b8` | **VERIFIED Untouched** |
| `results/pruning/mobilenetv2_pruning_metrics.json` | Pruning Metrics JSON | 11,236 | `0bb4c7b3bbda61be397a1d807656b3289d4cd3e2512426ba3784052ca09d7ed8` | **VERIFIED Untouched** |
| `results/pruning/mobilenetv2_pruning_history.csv` | Pruning Training History | 1,220 | `81e5de51f17220a690673ea26e3e74669b55018b099bde48dbecd7a43cb0661a` | **VERIFIED Untouched** |
| `results/int8/mobilenetv2_int8_verification_report.md` | Original INT8 Report | 7,710 | `192c6cd6dd36337e2a832c92412def49da6134e056a4599cea52bace33e8a23e` | **VERIFIED Untouched** |
| `results/int8/mobilenetv2_int8_metrics.json` | INT8 Metrics JSON | 4,234 | `15984d2f24a27a41c827ddcbbd9ba587c0a588769746b69a6e3dd639b1e8f08c` | **VERIFIED Untouched** |

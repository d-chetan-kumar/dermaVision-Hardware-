# DermaVision: MobileNetV2 Gradual Weight Pruning Report

**Date of Execution:** 2026-11-10 09:29:42  
**Source Checkpoint:** `models/best_mobilenetv2.keras`  
**Pruned Checkpoint:** `models/best_mobilenetv2_pruned.keras`  
**Hardware Evaluated:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (AMD64) on Windows 11  
**Framework Versions:** TensorFlow 2.21.0, Keras 3.15.0, TFMOT 0.8.1  
**Status:** Pruning Fine-Tuning & Evaluation Complete — All Constraints Honored

---

## 1. Executive Summary

Gradual unstructured magnitude-based weight pruning was successfully applied to the original DermaVision MobileNetV2 model using TensorFlow Model Optimization Toolkit (`tfmot`).

Key results on the complete 2,489-image original validation partition:
- **Actual Measured Prunable Weight Sparsity:** Achieved **49.9971% sparsity** across all eligible weight-bearing kernels (**1,234,825 exact zeros** out of **2,469,792** elements across 35 `Conv2D` and 3 `Dense` layers; slightly below 50.0000% because step 1500 under `end_step=1560`, `frequency=100` corresponds to theoretical target 49.9971%).
- **Overall Model Sparsity:** Across the entire network (2,602,563 total parameters including unpruned `DepthwiseConv2D`, `BatchNormalization`, and biases), the model contains **1,234,825 zeros (47.4465% overall sparsity)**.
- **Top-1 Accuracy:** The Pruned Keras model (Epoch 5 checkpoint at target sparsity) achieves **83.49%** compared to **90.96%** for the original unpruned FP32 model (a delta of **-7.47 percentage points**).
- **Melanoma Sensitivity:** Melanoma recall shifted from **87.06%** to **73.45%** (delta: **-13.61 percentage points**).
- **Gzip Compression Footprint:** While uncompressed `.keras` size remains constant at **30.21 MB (31,673,383 bytes)** due to fixed dense array storage, gzip-9 compressed size drops from **26.92 MB (28,229,451 bytes)** down to **5.73 MB (6,009,286 bytes)** (**81.03% compression reduction** vs raw size; **78.71% smaller** than gzipped FP32), demonstrating large entropy savings from zero-valued weights.
- **Inference Latency Audit:** Measured single-image inference latency on CPU is **180.40 ms** (Pruned Keras) vs **131.77 ms** (Original FP32). Unstructured sparsity does not automatically speed up standard dense GEMM CPU runtimes without specialized sparse kernels (e.g., XNNPACK sparse or DeepSparse).

---

## 2. Pruning & Fine-Tuning Configuration

| Parameter | Configuration Setting | Verified Engineering Detail |
|---|---|---|
| **Base Checkpoint** | `models/best_mobilenetv2.keras` | Exact original FP32 model, preserved untouched |
| **Verified Model Head** | `GAP -> Dense(256, relu) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.1) -> Dense(3, softmax)` | Verified directly from saved layers (2,602,563 total params) |
| **Pruning Algorithm** | `tfmot.sparsity.keras.prune_low_magnitude` | Unstructured magnitude-based pruning |
| **Pruning Schedule** | `PolynomialDecay` | Sparsity ramp from `initial_sparsity=0.0` to `final_sparsity=0.50` |
| **Schedule Parameters** | Begin Step: 0, End Step: 1560, Frequency: 100 | Calculated from 312 batches/epoch x 5 epochs = 1,560 steps |
| **Eligible Layers Pruned** | `Conv2D` (35 layers) & `Dense` (3 layers) | Exactly 38 weight-bearing layers wrapped (2,469,792 params = 94.90%) |
| **Preserved Layers** | `DepthwiseConv2D` (17), `BatchNormalization` (52), `Dropout` (2), `GAP` (1), `InputLayer` (1), etc. | All 122 non-eligible / unsupported layers preserved without pruning wrappers |
| **Fine-Tuning Dataset** | Original 9,956 training images | Partitioned from `ISIC_2019_Training_GroundTruth.csv` (`seed=42`), shuffled (`buffer_size=9956`) |
| **Validation Dataset** | Original 2,489 validation images | Zero data leakage; no validation/test data used for gradient updates |
| **Fine-Tuning Epochs** | 5 epochs | Monitored `val_accuracy` and `val_loss` with `UpdatePruningStep` callback |
| **Optimizer & LR** | Adam, Learning Rate = `1e-5` | Controlled low-LR fine-tuning |
| **Loss Function** | `sparse_categorical_crossentropy` | Class mapping: `0: BCC`, `1: Melanoma`, `2: Nevus` |
| **Batch Size** | 32 | Preprocessing: RGB 224 x 224, `pixel / 255.0` |

---

## 3. Pruning Fine-Tuning Training History & Checkpoint Selection

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Prunable Kernel Sparsity | Overall Model Sparsity | Duration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0114 | 99.70% | 0.5934 | 90.60% | **23.6541%** | 22.4474% | 524.2s |
| 2 | 0.0150 | 99.53% | 0.7854 | 84.61% | **38.3476%** | 36.3913% | 500.6s |
| 3 | 0.0184 | 99.40% | 1.8897 | 68.38% | **46.2134%** | 43.8558% | 645.4s |
| 4 | 0.0265 | 99.05% | 0.6347 | 81.48% | **49.3854%** | 46.8660% | 370.5s |
| 5 | 0.0344 | 99.04% | 0.6520 | 83.49% | **49.9971%** | 47.4465% | 427.1s |


**Scientific Observation on Training Trajectory & Checkpoint Selection:**
- At **Epoch 1**, validation accuracy remained high (**90.60%**), because the cubic `PolynomialDecay` schedule had only pruned **23.6541%** of prunable weights (`step 300`).
- During **Epochs 2–3**, aggressive sparsity increases (to **38.35%** and **46.21%**) temporarily disrupted MobileNetV2's compact bottleneck features and BatchNormalization statistics, dipping validation accuracy to **68.38%** at Epoch 3.
- During **Epochs 4–5**, as sparsity plateaued at **49.39%** and **49.9971%**, the network recovered substantially to **81.48%** (Epoch 4) and **83.49%** (Epoch 5).
- Rather than selecting the minimally pruned Epoch 1 checkpoint (~24% sparsity), the **Epoch 5 checkpoint** was selected so that the resulting model (`models/best_mobilenetv2_pruned.keras`) genuinely meets the **~50% target sparsity** specification while maximizing recovered validation accuracy at target sparsity. Note that extending fine-tuning past 5 epochs or adding constant-sparsity recovery epochs at `1e-5` would typically be needed for MobileNetV2's compact 1x1 convolutions to fully regain baseline 90%+ accuracy.

---

## 4. Actual Measured Sparsity Audit

Element-wise zero inspection conducted directly on the stripped & reloaded `models/best_mobilenetv2_pruned.keras`:

| Weight Component | Layer Count | Zero Weight Count | Total Elements | Measured Sparsity (%) | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Conv2D Kernels** | 35 | 1,062,706 | 2,125,536 | **49.9971%** | Pruned via `PolynomialDecay` |
| **Dense Kernels** | 3 | 172,119 | 344,256 | **49.9974%** | Pruned via `PolynomialDecay` |
| **Total Eligible Prunable Weights** | **38** | **1,234,825** | **2,469,792** | **49.9971%** | **~50% Target Achieved** |
| `DepthwiseConv2D` Kernels | 17 | 0 | 64,224 | 0.0000% | Preserved Unpruned (TFMOT Registry) |
| `BatchNormalization` Weights | 52 | 0 | 68,224 | 0.0000% | Preserved Unpruned |
| Biases (`Dense` layers) | 3 | 0 | 323 | 0.0000% | Preserved Unpruned |
| **Total Network Parameters** | **160 layers** | **1,234,825** | **2,602,563** | **47.4465%** | **Complete Model Footprint** |

---

## 5. Validation Metrics Comparison (2,489 Images)

### Overall Classification Performance

| Metric | Original FP32 Keras | Pruned Keras (49.9971% Kernel Sparsity) | Delta (Pruned vs FP32) |
|---|:---:|:---:|:---:|
| **Overall Accuracy** | **90.96%** | **83.49%** | **-7.47 pp** |
| **Macro Precision** | 91.11% | 84.84% | -6.27 pp |
| **Macro Recall** | 91.45% | 83.33% | -8.12 pp |
| **Macro F1-Score** | 91.27% | 83.55% | -7.72 pp |

### Detailed Per-Class Breakdown

| Class | Model | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|:---:|:---:|:---:|:---:|
| **BCC (0)** | Original FP32 | 93.04% | 96.54% | 94.76% | 665 |
| | **Pruned Keras** | **89.25%** | **82.41%** | **85.69%** | 665 |
| **Melanoma (1)** | Original FP32 | 89.03% | 87.06% | 88.03% | 904 |
| | **Pruned Keras** | **87.95%** | **73.45%** | **80.05%** | 904 |
| **Nevus (2)** | Original FP32 | 91.26% | 90.76% | 91.01% | 920 |
| | **Pruned Keras** | **77.32%** | **94.13%** | **84.90%** | 920 |

---

## 6. Confusion Matrices

### A. Original FP32 Keras
```
                Predicted BCC    Predicted Melanoma    Predicted Nevus
True BCC             642              21                    2
True Melanoma        39               787                   78
True Nevus           9                76                    835
```

### B. Pruned Keras (49.9971% Kernel Sparsity)
```
                Predicted BCC    Predicted Melanoma    Predicted Nevus
True BCC             548              39                    78
True Melanoma        64               664                   176
True Nevus           2                52                    866
```

---

## 7. Model Size, Compression & Inference Speed Comparison

- **Benchmarking Protocol:** Batch size = 1 (single image inference).
- **Execution:** 50 warm-up runs followed by 200 consecutive timed runs using `time.perf_counter()`.
- **Hardware Platform:** `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (AMD64) on Windows 11`

| Model Variant | Checkpoint Path | Raw Size (MB / Bytes) | Gzip-9 Size (MB / Bytes) | Parameter Sparsity | Mean Latency (ms) | Std Dev (ms) | Throughput (FPS) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original FP32 Keras** | `models/best_mobilenetv2.keras` | **30.21 MB** (`31,673,490` B) | **26.92 MB** (`28,229,451` B) | 0.0000% (Overall: 0.0000%) | **131.77 ms** | $\pm$ 17.16 ms | 7.6 FPS |
| **Dynamic-Range TFLite** | `models/best_mobilenetv2.tflite` | **2.74 MB** (`2,877,224` B) | **2.39 MB** (`2,501,547` B) | Unpruned (Quantized Weights) | **38.17 ms** | $\pm$ 10.91 ms | 26.2 FPS |
| **Full Integer INT8 TFLite** | `models/best_mobilenetv2_full_int8.tflite` | **2.94 MB** (`3,083,384` B) | **2.44 MB** (`2,553,496` B) | Unpruned (Full INT8) | **6.63 ms** | $\pm$ 2.29 ms | **150.8 FPS** |
| **Pruned Keras** | `models/best_mobilenetv2_pruned.keras` | **30.21 MB** (`31,673,383` B) | **5.73 MB** (`6,009,286` B) | **49.9971% (Overall: 47.4465%)** | **180.40 ms** | $\pm$ 34.64 ms | 5.5 FPS |

### Architectural Insights on Latency & Sparsity
1. **Unstructured Sparsity vs. Standard Dense Execution:** Standard BLAS and TensorFlow CPU FP32 GEMM kernels operate uniformly over dense tensor layouts. Because pruned zero elements remain stored in standard dense tensors after `strip_pruning()`, ordinary Keras/CPU inference executes the exact same arithmetic instructions (`0.0 * x = 0.0`) unless exported to a sparse-aware inference runtime (such as XNNPACK sparse kernels in TFLite or Neural Magic DeepSparse).
2. **Entropy Compression Benefit:** While the raw `.keras` file size matches the FP32 baseline (`~30.21 MB`, including optimizer state/architecture ZIP payload), `gzip -9` compression clearly reveals the impact of 1.23M zero weights, reducing the compressed model archive substantially compared to the dense FP32 model.
3. **Comparison with Post-Training INT8 Quantization:** Without changing inference runtimes, Full Integer INT8 TFLite quantization achieves far greater speedups (**~16x faster** CPU latency) and immediate uncompressed disk reduction (`2.94 MB`) with virtually zero accuracy loss (`90.72%` vs `90.96%`).

---

## 8. Preservation Compliance Audit

All pre-existing project artifacts remain untouched and fully preserved:
- `models/best_mobilenetv2.keras`: Unchanged (MD5/SHA preserved)
- `models/best_mobilenetv2.tflite`: Unchanged
- `models/best_mobilenetv2_full_int8.tflite`: Unchanged
- `evaluation.py`: Unchanged
- Frontend/Streamlit applications: Unchanged

New Artifacts Created:
- Stripped Model: `models/best_mobilenetv2_pruned.keras`
- Training Pipeline: `prune_and_train_mobilenetv2.py`
- Evaluation Pipeline: `evaluate_pruned_mobilenetv2.py`
- History Log: `results/pruning/mobilenetv2_pruning_history.csv`
- Verification Report: `results/pruning/mobilenetv2_pruning_report.md`
- Metrics JSON: `results/pruning/mobilenetv2_pruning_metrics.json`

# DermaVision: MobileNetV2 Full Integer INT8 Quantization Verification Report

**Date of Execution:** 2026-11-10 08:18:17  
**Target Checkpoint:** `models/best_mobilenetv2.keras`  
**Output Checkpoint:** `models/best_mobilenetv2_full_int8.tflite`  
**Hardware Evaluated:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (AMD64) on Windows 11  
**TensorFlow Version:** 2.21.0  
**Status:** Verification Complete — All Strict Preservation Constraints Honored

---

## 1. Executive Summary

Full Integer INT8 post-training quantization was successfully applied to the original DermaVision MobileNetV2 model without any fine-tuning or retraining. 

Key results on the complete 2,489-image original validation partition:
- **Top-1 Accuracy:** The Full INT8 model achieves **90.72%** compared to **90.96%** for the original FP32 Keras model (a difference of **-0.24 percentage points**).
- **Melanoma Sensitivity:** Melanoma recall shifted from **87.06%** (FP32) to **86.73%** (Full INT8).
- **Storage Footprint:** Model size dropped from **30.21 MB** to **2.94 MB** (**90.3% reduction**).
- **Latency Acceleration:** Average per-image inference latency decreased from **269.56 ms** (FP32) down to **16.35 ms** (Full INT8), delivering a **16.5x CPU speedup**.
- **Quantization Integrity:** Both input and output tensors are strictly **INT8** (`int8`), with quantization scales and zero-points correctly mapped.

---

## 2. Quantization & Calibration Configuration

| Parameter | Configuration Setting | Verification Note |
|---|---|---|
| **Base Model** | `models/best_mobilenetv2.keras` | Exact original checkpoint, untouched |
| **Quantization Scheme** | Full Integer Quantization (PTQ) | Integer arithmetic across all conv, dense, and pooling layers |
| **Optimization Flags** | `[tf.lite.Optimize.DEFAULT]` | Enabled integer kernel conversions |
| **Supported Operations** | `[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]` | Strictly integer-only builtins |
| **Inference Input Type** | `tf.int8` | Requires quantized int8 input in [-128, 127] |
| **Inference Output Type** | `tf.int8` | Returns quantized int8 output in [-128, 127] |
| **Calibration Set Size** | 250 images | Sampled from `train_df` with `random_state=42` |
| **Calibration Data Isolation** | **Strictly Training Set Only** | **0 samples from validation or test sets** |
| **Calibration Class Distribution** | Nevus: 104, Mel: 78, BCC: 68 | Reflects training prior distribution |
| **Calibration Preprocessing** | Resize to 224 x 224 RGB, pixel / 255.0 | Matches original training preprocessing |

---

## 3. Tensor Details & Quantization Parameters

### Input Tensor
- **Tensor Name:** `serving_default_input_layer_1:0`
- **Tensor Shape:** `[1, 224, 224, 3]`
- **Data Type:** `numpy.int8` (`int8`)
- **Scale (S_in):** `0.003921568859368563` (exact `1 / 255.0`)
- **Zero Point (Z_in):** `-128`
- **Input Quantization Formula:**
  `q_in = clip(round(x_fp32 / 0.00392157 + (-128)), -128, 127)`

### Output Tensor
- **Tensor Name:** `StatefulPartitionedCall_1:0`
- **Tensor Shape:** `[1, 3]`
- **Data Type:** `numpy.int8` (`int8`)
- **Scale (S_out):** `0.00390625` (exact `1 / 256.0`)
- **Zero Point (Z_out):** `-128`
- **Output Dequantization Formula:**
  `p_fp32 = (q_out - (-128)) * 0.00390625`
- **Sum of Probabilities:** Raw dequantized probabilities have a mean sum of **0.9967** across all 2,489 validation images, confirming that 8-bit integer Softmax approximates unit probability distribution.

---

## 4. Evaluation Metrics Comparison

Evaluated on the exact same **2,489 original validation images** (BCC: 665, Melanoma: 904, Nevus: 920):

### Overall Classification Metrics

| Metric | Original Keras FP32 | Existing Dynamic-Range TFLite | New Full Integer INT8 TFLite | Delta (INT8 vs FP32) |
|---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **90.96%** | 90.88% | **90.72%** | **-0.24 pp** |
| **Macro Precision** | 91.11% | 91.09% | 90.99% | -0.12 pp |
| **Macro Recall** | 91.45% | 91.36% | 91.07% | -0.39 pp |
| **Macro F1-Score** | 91.27% | 91.20% | 91.03% | -0.24 pp |

### Per-Class Detailed Performance

| Class | Model | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|:---:|:---:|:---:|:---:|
| **BCC (0)** | Original FP32 | 93.04% | 96.54% | 94.76% | 665 |
| | Dynamic Range | 93.58% | 96.39% | 94.96% | 665 |
| | **Full INT8** | **94.03%** | **94.74%** | **94.38%** | 665 |
| **Melanoma (1)** | Original FP32 | 89.03% | 87.06% | 88.03% | 904 |
| | Dynamic Range | 89.79% | 85.62% | 87.66% | 904 |
| | **Full INT8** | **88.49%** | **86.73%** | **87.60%** | 904 |
| **Nevus (2)** | Original FP32 | 91.26% | 90.76% | 91.01% | 920 |
| | Dynamic Range | 89.92% | 92.07% | 90.98% | 920 |
| | **Full INT8** | **90.46%** | **91.74%** | **91.10%** | 920 |

---

## 5. Confusion Matrices

### A. Original Keras FP32
```
                Predicted BCC    Predicted Melanoma    Predicted Nevus
True BCC             642              21                    2
True Melanoma        39               787                   78
True Nevus           9                76                    835
```

### B. New Full Integer INT8 TFLite
```
                Predicted BCC    Predicted Melanoma    Predicted Nevus
True BCC             630              32                    3
True Melanoma        34               784                   86
True Nevus           6                70                    844
```

---

## 6. Model Size & Inference Latency Benchmark

- **Benchmarking Protocol:** Batch size = 1 (simulating single dermoscopic image acquisition on an edge medical device).
- **Protocol Details:** 50 warm-up runs followed by 200 precisely timed consecutive inference runs using `time.perf_counter()`.
- **Latency Accounting:** Full INT8 timing includes full end-to-end quantization of input image and dequantization of output logits.

| Model Variant | File Path | File Size (MB) | Compression vs FP32 | Mean Latency per Image (ms) | Std Dev (ms) | Throughput (FPS) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Original Keras FP32** | `models/best_mobilenetv2.keras` | **30.21 MB** | Baseline (0.0%) | **269.56 ms** | $\pm$ 83.04 ms | 3.7 img/s |
| **Dynamic-Range TFLite** | `models/best_mobilenetv2.tflite` | **2.74 MB** | 90.9% reduction | **73.74 ms** | $\pm$ 9.22 ms | 13.6 img/s |
| **Full Integer INT8 TFLite** | `models/best_mobilenetv2_full_int8.tflite` | **2.94 MB** | **90.3% reduction** | **16.35 ms** | $\pm$ 2.43 ms | **61.2 img/s** |

---

## 7. Unsupported Operations & Conversion Warnings Audit

- **Unsupported Operations:** **None**. All MobileNetV2 building blocks (`Conv2D`, `DepthwiseConv2D`, `Add`, `ReLU6`, `GlobalAveragePooling2D`, `Dense`, `Softmax`) were mapped to standard `TFLITE_BUILTINS_INT8` integer kernels.
- **Conversion Warnings:** No fallback to floating point operators occurred. The model is 100% executable on integer-only NPUs and DSP accelerators.
- **Dequantization Stability:** Output probabilities sum to $\approx 1.0$ without numerical underflow or overflow.

---

## 8. Artifact Verification & Preservation Status

All pre-existing project artifacts remain untouched:
- `models/best_mobilenetv2.keras`: Unchanged (MD5/SHA preserved)
- `models/best_mobilenetv2.tflite`: Unchanged
- `evaluation.py`: Unchanged
- Frontend/Streamlit applications: Unchanged

New Artifacts Created:
- Model: `models/best_mobilenetv2_full_int8.tflite`
- Pipeline script: `quantize_and_evaluate_int8.py`
- Verification report: `results/int8/mobilenetv2_int8_verification_report.md`
- Raw metrics JSON: `results/int8/mobilenetv2_int8_metrics.json`

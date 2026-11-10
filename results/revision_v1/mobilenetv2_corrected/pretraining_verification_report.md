# DermaVision Revision V1.1: MobileNetV2 Corrected Pipeline Pre-Training Verification Report

**Author:** DermaVision Research & Engineering Team  
**Date:** November 9, 2026  
**Status:** Implementation Complete, Verified & Awaiting User Training Authorization  
**Target Environment:** Revision V1 Lesion-Grouped (`train.csv`: 8,712 | `val.csv`: 1,875 | `test.csv`: Quarantined)  
**Implementation Script:** `train_mobilenetv2_revision_v1_1.py`  

---

## 1. Executive Summary

In accordance with user authorization and diagnostic findings, the **Revision V1.1 Corrected MobileNetV2 Training Pipeline** has been created, verified, and smoke-tested. 

### Key Corrections Implemented:
1. **Data Shuffling Fix:** Eliminated the pseudo-sequential class blocking defect by combining **in-memory DataFrame shuffling** (`train_df.sample(frac=1.0, random_state=42)`) with a **full-dataset tf.data buffer** (`buffer_size=8,712`). Every training batch is now a truly stochastic, independent and identically distributed (IID) mixture of all 3 diagnostic classes.
2. **Two-Stage Fine-Tuning:**
   * **Stage 1 (Head Warmup, Epochs 1–10):** All 154 backbone layers are frozen. Only the classification head (6 weight tensors) is trained at $\text{lr}=5 \times 10^{-4}$ using Adam, eliminating destructive early gradient backpropagation into ImageNet weights.
   * **Stage 2 (Backbone Fine-Tuning, Epochs 11–50):** The backbone is un-frozen, but **all 52 BatchNormalization layers are strictly frozen** (`layer.trainable = False`), locking ImageNet moving mean/variance. Training proceeds at $\text{lr}=1 \times 10^{-5}$ with `ReduceLROnPlateau`.
3. **Clinical Data Augmentation (Training Only):** Incorporated label-preserving transformations: horizontal and vertical flips, continuous rotation up to $\pm 45^\circ$, and mild zoom up to $\pm 10\%$. Validation remains strictly deterministic.
4. **Preserved Baseline Invariants:** Retains $[0.0, 1.0]$ input scaling (`/ 255.0`), unweighted cross-entropy, exact 3-class mapping (`BCC=0, Mel=1, Nevus=2`), and zero test data access.

---

## 2. Batch Distribution Audit & Shuffling Verification

An audit of the corrected pipeline was executed on `data/splits/revision_v1/train.csv` across 10 consecutive batches ($N = 64$ per batch):

```
+──────────────+──────────────+─────────────────+─────────────────+─────────────────+──────────────────────────+
| Batch Index  | Batch Size   | BCC (Class 0)   | Mel (Class 1)   | Nevus (Class 2) | All 3 Classes Present?   |
+──────────────+──────────────+─────────────────+─────────────────+─────────────────+──────────────────────────+
| Batch 00     | 64           | 14 (21.88%)     | 30 (46.88%)     | 20 (31.25%)     | YES                      |
| Batch 01     | 64           | 23 (35.94%)     | 20 (31.25%)     | 21 (32.81%)     | YES                      |
| Batch 02     | 64           | 11 (17.19%)     | 31 (48.44%)     | 22 (34.38%)     | YES                      |
| Batch 03     | 64           | 20 (31.25%)     | 19 (29.69%)     | 25 (39.06%)     | YES                      |
| Batch 04     | 64           | 12 (18.75%)     | 21 (32.81%)     | 31 (48.44%)     | YES                      |
| Batch 05     | 64           | 24 (37.50%)     | 17 (26.56%)     | 23 (35.94%)     | YES                      |
| Batch 06     | 64           | 19 (29.69%)     | 15 (23.44%)     | 30 (46.88%)     | YES                      |
| Batch 07     | 64           | 13 (20.31%)     | 18 (28.12%)     | 33 (51.56%)     | YES                      |
| Batch 08     | 64           | 17 (26.56%)     | 18 (28.12%)     | 29 (45.31%)     | YES                      |
| Batch 09     | 64           | 11 (17.19%)     | 21 (32.81%)     | 32 (50.00%)     | YES                      |
+──────────────+──────────────+─────────────────+─────────────────+─────────────────+──────────────────────────+
| Empirical 10-Batch Mean     | 16.4 (25.6%)    | 21.0 (32.8%)    | 26.6 (41.6%)    | 100% of batches mixed    |
| Theoretical Expectation     | 17.1 (26.8%)    | 23.2 (36.2%)    | 23.7 (37.0%)    | Expected Class Balance   |
+──────────────+──────────────+─────────────────+─────────────────+─────────────────+──────────────────────────+
```

* **Comparison with Defective Pipeline:** In the previous uncorrected run, Batches 0–33 were **100% pure Melanoma** (64/64), followed by pure Nevus, followed by pure BCC. In Revision V1.1, **100% of batches contain all 3 classes in healthy, balanced proportions**.
* The full audit artifact is saved at `results/revision_v1/mobilenetv2_corrected/batch_distribution_audit.json`.

---

## 3. Two-Stage Architectural & Trainability Verification

```
STAGE 1: Classification Head Warmup (Epochs 1 – 10)
┌──────────────────────────────────────────────────────────┐
│ MobileNetV2 Backbone (154 layers)                        │ ──► TRAINABLE: FALSE (Frozen)
│ Pretrained ImageNet Weights (1.00 224)                   │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ GlobalAveragePooling2D                                   │
│ Dense (256, ReLU) ──► Dropout (0.2)                      │ ──► TRAINABLE: TRUE (6 Weight Tensors)
│ Dense (64, ReLU)  ──► Dropout (0.1)                      │     Optimizer: Adam (lr = 0.0005)
│ Dense (3, Softmax)                                       │
└──────────────────────────────────────────────────────────┘

STAGE 2: Backbone Fine-Tuning with Frozen BatchNorm (Epochs 11 – 50)
┌──────────────────────────────────────────────────────────┐
│ 52 BatchNormalization Layers                             │ ──► TRAINABLE: FALSE (Frozen ImageNet stats)
│ 52 Convolutional / Depthwise Layers                      │ ──► TRAINABLE: TRUE (52 Conv Kernels)
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Classification Head (3 Dense Layers)                     │ ──► TRAINABLE: TRUE (6 Weight Tensors)
│ Total Trainable Tensors: 52 + 6 = 58 Tensors             │     Optimizer: Adam (lr = 0.00001)
│ LR Scheduler: ReduceLROnPlateau (patience=4, factor=0.5) │
└──────────────────────────────────────────────────────────┘
```

### Smoke Test Verification Results
* **Stage 1 Parameter Check:** Exactly **6 trainable weight tensors** in the model (`head_dense_1/kernel`, `head_dense_1/bias`, `head_dense_2/kernel`, `head_dense_2/bias`, `head_predictions/kernel`, `head_predictions/bias`). Forward and backward gradient updates verified without errors ($\text{Loss} = 0.7141$).
* **Stage 2 Parameter Check:** Exactly **52 BatchNormalization layers frozen**. Total trainable tensors: **58** ($52$ backbone conv kernels $+ 6$ classification head tensors). Forward and backward gradient pass verified ($\text{Loss} = 0.7503$).
* **Metrics Callback:** `ComprehensiveMetricsCallback` verified. Calculated and logged Overall Accuracy, Macro Precision, Macro Recall, Macro F1, BCC Recall, Melanoma Recall, and Nevus Recall to CSV.

---

## 4. Manifest Provenance & Test Isolation

* **Training Manifest:** `data/splits/revision_v1/train.csv` (8,712 samples)
  * SHA-256: `f8a31b9d42fffe8beec85994f8a3ddbc0cfa3198038b34003d15dc012971ba48`
* **Validation Manifest:** `data/splits/revision_v1/val.csv` (1,875 samples)
  * SHA-256: `75da1423e640f0c0587fb878b209e7c53648fa34c11b069d12a67e20ea2189d9`
* **Test Isolation Security Check:**
  * When `train_mobilenetv2_revision_v1_1.py` was invoked with `--train_manifest data/splits/revision_v1/test.csv`, it immediately halted with:
    `SECURITY VIOLATION: Test manifest 'test.csv' detected in training pipeline! Test data must remain strictly isolated until final frozen model evaluation.`
  * Return Code: `1` (Verified).

---

## 5. Output Locations for Revision V1.1

All outputs are completely segregated from the legacy runs and the baseline V1 run:

* **Model Checkpoints:** `models/revision_v1/mobilenetv2_corrected/`
  * `best_mobilenetv2.keras` (monitored on `val_accuracy`)
  * `final_mobilenetv2.keras` (saved at Epoch 50)
* **Results and Logs:** `results/revision_v1/mobilenetv2_corrected/`
  * `training_history.csv` (epoch-wise metric records)
  * `training_config.json` (complete configuration snapshot)
  * `batch_distribution_audit.json` (IID batch audit)
  * `training_summary.json` (best metrics, duration, checkpoint paths)

---

## 6. Readiness Declaration

The pipeline implementation `train_mobilenetv2_revision_v1_1.py` is complete, structurally sound, and fully verified. 

**No training has been launched.** The background baseline task (`task-521`) continues uninterrupted. We await your explicit instructions before initiating full training of the corrected pipeline.


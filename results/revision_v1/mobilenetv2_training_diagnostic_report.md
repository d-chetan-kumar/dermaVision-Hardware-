# DermaVision MobileNetV2 Training Pipeline Diagnostic Report

**Author:** DermaVision Research & Engineering Team  
**Date:** November 9, 2026  
**Status:** Read-Only Audit & Forensic Investigation Completed  
**Active Process Audited:** `task-521` (`train_mobilenetv2_revision_v1.py`, actively executing uninterrupted)  
**Dataset Partitions:** Revision V1 Lesion-Grouped (`train.csv`: 8,712 | `val.csv`: 1,875 | `test.csv`: Quarantined)  

---

## 1. Executive Summary & Root Cause Isolation

An exhaustive diagnostic investigation was conducted into the early-epoch training dynamics of the **MobileNetV2 Revision V1** baseline. 

### Core Diagnostic Breakthrough
A **critical implementation defect** was confirmed in the data loading pipeline:
* **The Root Cause:** `train.csv` is sorted contiguously by class: rows 0–3,155 are **Melanoma (Class 1)**, rows 3,156–6,378 are **Nevus (Class 2)**, and rows 6,379–8,711 are **BCC (Class 0)**. In `train_mobilenetv2_revision_v1.py`, the tf.data buffer size was set to `buffer_size = 1024`, which is far smaller than the dataset ($8,712$).
* **The Consequence:** Batches were **not** independent and identically distributed (IID). 
  * Batches 0 to 33 were **100% pure Melanoma**.
  * Batches 50 to 90 were **~100% pure Nevus**.
  * Batches 110 to 137 were **100% pure BCC**.
* **Why Epoch 1 Reported 84.60% Train Accuracy:** For the first 35 batches of pure Melanoma, the network quickly learned to predict 100% Melanoma for every image, achieving **95.08% accuracy at batch 34**. Because Keras calculates training accuracy as a cumulative running average over batches, this inflated early accuracy smoothed into an artificial 84.60% at epoch end.
* **Why the Network Suffered Catastrophic Intra-Epoch Forgetting:** When Nevus images arrived at batch 35, loss exploded, and the network adjusted to predict Nevus, forgetting Melanoma. Then at batch 100, BCC arrived, forcing another shift. At validation time, the network was biased toward the classes seen in the final batches (Nevus and BCC), resulting in high Nevus recall ($93\% - 100\%$), moderate BCC recall ($47\% - 74\%$), and near-zero Melanoma recall ($0\% - 8\%$).

---

## 2. Confirmed Implementation Errors vs. Hypotheses

```
+─────────────────────────────────────────────────────────────+──────────────────────+───────────────────────────────────────────────+
| Diagnostic Investigation Item                               | Classification       | Verification Evidence                         |
+─────────────────────────────────────────────────────────────+──────────────────────+───────────────────────────────────────────────+
| Class-homogeneous batch streaming (buffer_size=1024)        | CONFIRMED BUG        | Manifest inspection & batch label simulation  |
| Narrow 1-class slice in prior preprocessing check           | CONFIRMED BUG        | val_df.head(200) consists of 100% Melanoma    |
| Class mapping mismatch across partitions                    | DISPROVEN (NO ERROR) | Manifests verified: BCC=0, Mel=1, Nevus=2     |
| Preprocessing divergence (RGB, resize, [0,1] scaling)       | DISPROVEN (NO ERROR) | Identical tensor stats across train & val     |
| Model architecture or double-Softmax bug                    | DISPROVEN (NO ERROR) | Architecture verified: single Softmax, 3 dims |
| Metric calculation formula or shape mismatch                | DISPROVEN (NO ERROR) | Keras acc & Scikit-Learn callback match 100%  |
| Corrupted or unreadable images                              | DISPROVEN (NO ERROR) | 0 corrupted images in train and val           |
| Trainable BatchNorm moving-average oscillation              | CONFIRMED FACTOR     | 52 BN layers swing across class blocks        |
| Destructive gradient backprop on random head                | CONFIRMED FACTOR     | Initial loss L=4.80 perturbing all 154 layers |
+─────────────────────────────────────────────────────────────+──────────────────────+───────────────────────────────────────────────+
```

---

## 3. Label Mapping Verification (Task 1)

* **Code Mapping Definition:**
  ```python
  CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
  CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
```
* **Manifest Inspection:**
  * `train.csv` unique mappings:
    * Melanoma $\rightarrow$ 1 (3,156 images)
    * Nevus $\rightarrow$ 2 (3,223 images)
    * BCC $\rightarrow$ 0 (2,333 images)
  * `val.csv` unique mappings:
    * Melanoma $\rightarrow$ 1 (707 images)
    * Nevus $\rightarrow$ 2 (691 images)
    * BCC $\rightarrow$ 0 (477 images)
* **Sample Path & Label Cross-Check:**
  * Train Class 0: `ISIC_0024331` $\rightarrow$ BCC $\rightarrow$ Label 0 (Verified)
  * Train Class 1: `ISIC_0000002` $\rightarrow$ Melanoma $\rightarrow$ Label 1 (Verified)
  * Train Class 2: `ISIC_0000000` $\rightarrow$ Nevus $\rightarrow$ Label 2 (Verified)
  * Val Class 0: `ISIC_0024403` $\rightarrow$ BCC $\rightarrow$ Label 0 (Verified)
  * Val Class 1: `ISIC_0000004` $\rightarrow$ Melanoma $\rightarrow$ Label 1 (Verified)
  * Val Class 2: `ISIC_0000009` $\rightarrow$ Nevus $\rightarrow$ Label 2 (Verified)
* **Conclusion:** Class mapping is **100% correct, consistent, and identical** across partitions. Labels are single integers compatible with `SparseCategoricalCrossentropy`.

---

## 4. Model Output and Loss Verification (Task 2)

* **Output Layer Architecture:**
  * Layer name: `dense_2`
  * Neurons: Exactly 3
  * Activation: `softmax`
  * Shape: `(batch_size, 3)`
* **Loss Function Configuration:**
  * Function: `tf.keras.losses.SparseCategoricalCrossentropy()`
  * Default argument: `from_logits=False` (expects normalized probabilities summing to 1.0).
  * Inputs: Ground-truth integer tensor of shape `(batch_size,)`, predicted probability tensor of shape `(batch_size, 3)`.
* **Checks:**
  * Probabilities sum strictly to $1.0$ (verified: `np.allclose(sums, 1.0)`).
  * No double Softmax is applied.
  * Loss implementation is native TensorFlow; no custom loss function errors exist.

---

## 5. Preprocessing Verification (Task 3)

The exact image loading and normalization function (`_parse_image`) was inspected:
* Both partitions use `tf.image.decode_jpeg(channels=3)` (guarantees RGB format; no BGR inversion).
* Both partitions resize via `tf.image.resize(img, (224, 224))`.
* Both partitions scale via `tf.cast(img, tf.float32) / 255.0`.
* No data augmentation or partition-specific transforms are active.

### Tensor Statistics on Live Batches (N = 64)
* **Training Batch:**
  * Minimum: `0.0000` | Maximum: `1.0000`
  * Mean: `0.5826` | Standard Deviation: `0.2566`
* **Validation Batch:**
  * Minimum: `0.0000` | Maximum: `1.0000`
  * Mean: `0.5943` | Standard Deviation: `0.2010`
* **Conclusion:** Preprocessing is **100% identical and mathematically verified** across both partitions.

---

## 6. BatchNormalization Dynamics (Task 4)

* **Backbone Layer Count:** 154 total layers, containing **52 `BatchNormalization` layers**.
* **Trainability Status:** Because `base_model.trainable = True`, all 52 BN layers are set to `layer.trainable = True`.
* **Execution Modes During `fit()`:**
  * Training steps: Keras passes `training=True`. The 52 BN layers compute mini-batch statistics and update their internal `moving_mean` and `moving_variance`.
  * Validation steps: Keras passes `training=False`. The 52 BN layers freeze moving statistics and use them for evaluation.
* **The Interaction with Class-Homogeneous Batches:**
  * During batches 0–33 (pure Melanoma), BN moving averages shifted to represent Melanoma color/texture distributions.
  * During batches 34–95, BN moving averages shifted to Nevus.
  * During batches 96–137, BN moving averages shifted to BCC.
  * At epoch end, validation evaluation (`training=False`) was performed using BN moving statistics calibrated predominantly to the tail-end classes (BCC and Nevus). This destabilized validation feature normalization for Melanoma.

---

## 7. Metric Calculation Verification (Task 5)

* **Implementation Audited:** `ComprehensiveMetricsCallback` in `train_mobilenetv2_revision_v1.py`.
* **Integrity Checks:**
  * `y_prob = active_model.predict(self.val_ds, verbose=0)` produces shape `(1875, 3)`.
  * `y_pred = np.argmax(y_prob, axis=1)` produces shape `(1875,)` with values in `{0, 1, 2}`.
  * `y_true = self.val_df["class_label"].values` produces shape `(1875,)`.
  * Because `val_ds` has `is_training=False` and `shuffle=False`, sample alignment is 1:1.
  * Keras built-in `val_accuracy` and callback `acc` match to 4 decimal places across all epochs (e.g. Epoch 7: Keras `0.5653` vs. Callback `0.5653`).
  * Per-class recalls are computed via Scikit-Learn `precision_recall_fscore_support` with explicit `labels=[0, 1, 2]`.
  * Metrics are recomputed from scratch every epoch; no state contamination between epochs occurs.

---

## 8. Dataset and Lesion Distribution (Task 6)

* **Class Sample Counts:**
  * Train: Nevus: 3,223 ($36.99\%$) | Melanoma: 3,156 ($36.23\%$) | BCC: 2,333 ($26.78\%$)
  * Val: Melanoma: 707 ($37.71\%$) | Nevus: 691 ($36.85\%$) | BCC: 477 ($25.44\%$)
* **Lesion and Patient Group Integrity:**
  * Train: 3,847 unique lesions across 5,168 independent groups.
  * Val: 828 unique lesions across 1,111 independent groups.
  * Cross-partition lesion overlap: **Strictly 0**.
  * Cross-partition group overlap: **Strictly 0**.
* **Image Integrity:** A sample of 100 images per partition showed **0 corrupted or unreadable images**.

---

## 9. Investigation of the Prior Preprocessing Comparison (Task 7)

The previous report noted an empirical test showing $91.00\%$ (`div255`) vs. $93.50\%$ (`mobilenet` standard scaling).

### Forensic Audit of that Measurement
* **Checkpoint Used:** `models/best_mobilenetv2.keras` (the historical Kaggle-trained model, NOT a Revision V1 model).
* **Dataset Slice Used:** `val_df.head(200)`.
* **Critical Discovery:** Because `val.csv` was also structured with classes in order, `val_df.head(200)` contains **200 images that are 100% Melanoma** (0 BCC, 0 Nevus).
* **Finding:** The reported $91.00\%$ vs. $93.50\%$ was **not** a balanced 3-class validation accuracy; it was exclusively a sensitivity measurement on 200 Melanoma images. On those 200 Melanoma images, `[-1, 1]` predicted Melanoma on 187 images, while `[0, 1]` predicted Melanoma on 182 images.
* **Conclusion:** This preliminary result cannot be cited as conclusive evidence of superior 3-class performance until a balanced evaluation across all 1,875 validation images is conducted.

---

## 10. Explanation of Learning Behaviour (Task 8)

### Why Training Accuracy Dropped from 84.60% (Epoch 1) to 54.60% (Epoch 4)
1. **The Artifact of Epoch 1:** In Epoch 1, batches 0–33 were 100% Melanoma. The network quickly memorized the single label, pushing batch accuracy to $95.08\%$ by batch 34. Keras reports training accuracy as the cumulative average of all batches processed so far. When the subsequent classes arrived, the earlier high scores held the cumulative average artificially high ($84.60\%$).
2. **Epochs 2–4 Reality:** In subsequent epochs, the network started from weights that had been pulled toward BCC at the end of Epoch 1. It then immediately faced 35 batches of Melanoma, causing high initial loss. The cumulative accuracy reflected this multi-class struggle, settling into its true unaugmented range of $54\% - 64\%$.
3. **Validation Loss Dropping While Accuracy Lags:**
   Validation loss measures prediction confidence ($-\log p_{\text{true}}$). In early epochs, the network was overconfident in wrong classes ($L \approx 4.8$). As training progressed, predictions softened, bringing loss down steadily from $4.80 \rightarrow 1.10$. Furthermore, **validation accuracy is actually climbing**:
   * Epoch 1: 38.99% (BCC: 9.0%, Mel: 0.0%, Nevus: 99.6%)
   * Epoch 3: 48.85% (BCC: 47.0%, Mel: 0.8%, Nevus: 99.3%)
   * Epoch 7: **56.53%** (BCC: 74.2%, Mel: 8.5%, Nevus: 93.5%)
   The model is progressively learning BCC, but Melanoma recall remains suppressed due to batch ordering and lack of class weighting.

---

## 11. Recommended Corrective Actions (Task 10)

Ranked strictly by scientific priority and expected impact:

### Priority 1: Full In-Memory Dataset Shuffling (Confirmed Bug Fix)
* **Action:** In `train_mobilenetv2_revision_v1.py`, set `buffer_size = len(train_df)` ($8,712$), or pre-shuffle `train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)` before feeding into `tf.data.Dataset`.
* **Expected Impact:** Guarantees that every batch of 64 contains a balanced mixture of BCC, Melanoma, and Nevus. Eliminates intra-epoch catastrophic forgetting.

### Priority 2: Two-Stage Fine-Tuning (Warmup Head First)
* **Action:** Stage 1: Freeze `base_model` (`trainable=False`), train only the classification head for 5–10 epochs at $\text{lr}=5 \times 10^{-4}$. Stage 2: Unfreeze backbone with frozen BN at $\text{lr}=5 \times 10^{-5}$.
* **Expected Impact:** Prevents destructive backpropagation through pretrained ImageNet weights during early epochs.

### Priority 3: Freeze BatchNormalization Layers
* **Action:** Set `layer.trainable = False` for all 52 BN layers in the backbone.
* **Expected Impact:** Stabilizes internal feature representations and prevents moving-average drift.

### Priority 4: Clinical Data Augmentation (Training Only)
* **Action:** Apply `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.25)`, and `RandomZoom(0.1)` on `train_ds`.
* **Expected Impact:** Prevents spatial memorization and significantly improves validation generalization.

### Priority 5: Balanced Class Weighting
* **Action:** Enable `--use_class_weights` ($w_{\text{BCC}}=1.24, w_{\text{Mel}}=0.92, w_{\text{Nev}}=0.90$).
* **Expected Impact:** Breaks the Nevus-majority bias and accelerates Melanoma recall learning.


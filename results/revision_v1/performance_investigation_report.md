# DermaVision Revision V1: MobileNetV2 Performance Investigation and Improvement Plan

**Author:** DermaVision Research Team  
**Date:** November 9, 2026  
**Status:** Audit & Proposal Document  
**Target Environment:** Revision V1 Lesion-Grouped (Train: 8,712 | Val: 1,875 | Test: 1,858 Quarantined)  

---

## 1. Executive Summary

This report investigates the observed early-epoch validation dynamics of the MobileNetV2 baseline training on the newly established **Revision V1 lesion-level grouped dataset**. 

Our previous audit proved that the historical 91.36% ensemble accuracy on the legacy split was inflated by **54.04% lesion-level overlap** (a +7.70% leakage disparity). On Revision V1, with zero lesion or patient leakage, the models face a genuine generalization challenge.

This report audits the ongoing baseline run (`task-521`), diagnoses the core technical factors limiting early convergence (destructive gradients on a random head, trainable BatchNorm on CPU batches, lack of rotational data augmentation, and unweighted majority-class collapse), and proposes a scientifically disciplined sequence of controlled experiments.

---

## 2. Live MobileNetV2 Training Audit (Task 1)

The currently running training task (`train_mobilenetv2_revision_v1.py`, background task `task-521`) was inspected non-intrusively via its process logs and `results/revision_v1/mobilenetv2/training_history.csv`.

### Empirical Progression Table (Completed Epochs 1 – 4)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Macro Prec | Macro Rec | Macro F1 | BCC Rec (0) | Mel Rec (1) | Nev Rec (2) | Learning Rate | Checkpoint Event |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.5478 | 84.60% | 4.8041 | **38.99%** | 28.95% | 36.19% | 23.58% | 9.01% | 0.00% | 99.57% | $1.0 \times 10^{-4}$ | Saved as initial best |
| **2** | 0.9334 | 64.12% | 4.2808 | **42.83%** | 63.18% | 41.14% | 30.34% | 23.27% | 0.28% | 99.86% | $1.0 \times 10^{-4}$ | Saved (improved) |
| **3** | 0.8249 | 64.06% | 3.0679 | **48.85%** | 65.12% | 49.03% | 38.09% | 46.96% | 0.85% | 99.28% | $1.0 \times 10^{-4}$ | Saved (improved) |
| **4** | 0.9551 | 54.60% | 2.3130 | **41.65%** | 56.55% | 39.24% | 28.48% | 15.30% | 2.40% | 100.00% | $1.0 \times 10^{-4}$ | Best remains Epoch 3 |
| **5** | *Running* | — | — | — | — | — | — | — | — | — | $1.0 \times 10^{-4}$ | In progress (batch 15/137) |

### Key Audit Observations
1. **Validation Loss is Steadily Dropping:** Validation loss has decreased every single epoch: **4.8041 $\rightarrow$ 4.2808 $\rightarrow$ 3.0679 $\rightarrow$ 2.3130**. This demonstrates that the model's posterior probability calibration is improving continuously.
2. **Learning Rate Scheduler:** `ReduceLROnPlateau` monitors `val_loss` with `patience=3`. Because `val_loss` has consistently improved, the scheduler has not triggered, and the learning rate remains at its initial value of $1.0 \times 10^{-4}$.
3. **Best Checkpoint on Disk:** Saved at **Epoch 3** with validation accuracy **48.85%** at `models/revision_v1/mobilenetv2/best_mobilenetv2.keras`.
4. **Dominant Pathology in Early Epochs:** **Nevus-biased collapse**. In Epoch 4, `nevus_recall` is **100.00%**, `bcc_recall` is **15.30%**, and `melanoma_recall` is **2.40%**. The network is currently relying on the benign majority class to minimize average cross-entropy.
5. **Stability:** The process has run continuously without errors or crashes.

---

## 3. Preprocessing Investigation (Task 2)

* **Historical Implementation:** Both `preprocessing.py` and `evaluation.py` explicitly divided images by $255.0$ to produce inputs in $[0.0, 1.0]$. The Keras `mobilenet_v2.preprocess_input` function was never imported or used.
* **ImageNet Pretrained Convention:** The ImageNet weights for MobileNetV2 were originally trained on $[-1.0, 1.0]$ ($(x / 127.5) - 1.0$). Feeding $[0.0, 1.0]$ introduces an input mean offset of $+0.5$, which forces the early convolutional layer (`Conv1`) to adapt its weights and biases to cancel this shift.
* **Empirical Check on Historical Checkpoint:** Testing 200 validation images on `best_mobilenetv2.keras`:
  * $[0.0, 1.0]$ input: **91.00% accuracy**
  * $[-1.0, 1.0]$ input: **93.50% accuracy** ($+2.50\%$ advantage)
* **Controlled Plan:** Experiment A will systematically compare `div255` vs `mobilenet` scaling on the exact same manifests, seed, and architecture.

---

## 4. BatchNormalization Investigation (Task 3)

The MobileNetV2 backbone contains **52 `BatchNormalization` layers**.

* **Current Baseline (Trainable BN):** All 52 BN layers are active and updating running mean and variance statistics during training.
  * *Disadvantages:* With a batch size of 64 on CPU, dermoscopic image batches have high visual variance compared to natural images. Overwriting ImageNet moving statistics with small-sample batch statistics destroys feature scaling in deep layers and degrades validation performance.
* **Alternative (Frozen BN):** Setting `layer.trainable = False` for all 52 BN layers while keeping convolutional weights trainable.
  * *Advantages:* Preserves ImageNet running mean and variance; acts as a strong regularizer; prevents internal covariate shift; recommended as best practice in transfer learning literature.
  * *Disadvantages:* Less adaptation to global dermoscopy illumination differences.
* **Controlled Plan:** Experiment B will evaluate Frozen BN vs. Trainable BN.

---

## 5. Fine-Tuning Strategy Investigation (Task 4)

* **Current Baseline (Strategy A):** Full-network fine-tuning from Epoch 1 at $\text{lr} = 1 \times 10^{-4}$.
  * *Issue:* The classification head (`Dense(256) -> Dense(64) -> Dense(3)`) is randomly initialized. During early epochs, massive loss gradients backpropagate through all 154 layers, causing destructive perturbation of pretrained weights.
* **Proposed Strategy B (Two-Stage Warmup Fine-Tuning):**
  * *Stage 1 (Head Warmup, Epochs 1–10):* Freeze the entire backbone (`base_model.trainable = False`). Train ONLY the classification head at $\text{lr} = 5 \times 10^{-4}$. The head learns coarse lesion boundaries while protecting backbone representations.
  * *Stage 2 (Backbone Fine-Tuning, Epochs 11–50):* Unfreeze the backbone (with frozen BN) and train end-to-end at a gentle learning rate ($\text{lr} = 1 \times 10^{-5}$ or $5 \times 10^{-5}$).
* **Proposed Strategy C (Partial Top-Block Fine-Tuning):**
  * Freeze the bottom 100 layers (general edge/texture extractors); fine-tune only the top 54 layers (blocks 14–16) at $\text{lr} = 5 \times 10^{-5}$.

---

## 6. Data Augmentation Investigation (Task 5)

Currently, **no data augmentation** is applied in `train_mobilenetv2_revision_v1.py`.

### Clinically Appropriate Augmentation Strategy
Skin lesions examined under a dermatoscope exhibit rotational and reflectional symmetry:
1. **Random Horizontal & Vertical Flips (`RandomFlip("horizontal_and_vertical")`):** 100% clinically valid; dermatoscopes have no fixed vertical axis.
2. **Random Rotation (`RandomRotation(factor=0.25)`, up to $\pm 90^\circ$):** Matches clinical practice where the clinician places the lens at arbitrary angles.
3. **Mild Zoom (`RandomZoom((-0.1, 0.1))`):** Simulates slight variations in lens distance.
4. **Subtle Contrast / Brightness Adjustments ($\pm 10\%$):** Simulates varying illumination intensity.

### Harmful Transformations (Strictly Prohibited)
* Extreme color/hue jittering: Alters melanin and erythema diagnostic colors.
* Severe shearing or elastic distortion: Corrupts border irregularity and asymmetry (the "A" and "B" in ABCDE criteria).
* Random cropping: Clips lesion margins, which are crucial for differential diagnosis.

*Crucial Guardrail:* Augmentation will be applied **strictly to training data**. Validation and test data remain 100% deterministic.

---

## 7. Class Imbalance Investigation (Task 6)

* **Class Distribution (Training):**
  * BCC: 2,333 ($26.78\%$)
  * Melanoma: 3,156 ($36.23\%$)
  * Nevus: 3,223 ($36.99\%$)
* **The Failure Mechanism:** Unweighted cross-entropy incentivizes the model to predict the visually common class (Nevus). In Epoch 4, Melanoma recall was just $2.40\%$, which is clinically unacceptable.
* **Balanced Weights:**
  $$w_{\text{BCC}} = 1.2447, \quad w_{\text{Melanoma}} = 0.9202, \quad w_{\text{Nevus}} = 0.9010$$
* **Melanoma Sensitivity Weighting:** In clinical practice, missing a melanoma has the highest cost. Experiment E will evaluate balanced weighting vs. sensitivity weighting ($w_{\text{Mel}} = 1.5$) to prioritize Melanoma recall and Macro F1 over simple overall accuracy.

---

## 8. DINOv2 Investigation and Improvement Plan (Task 7)

* **Current Baseline:** Frozen DINOv2 ViT-B/14 backbone with a single linear classification head (`nn.Linear(768, 3)`).
* **Limitations:** DINOv2 was pretrained on natural images (LVD-142M). A purely linear probe cannot adapt the underlying self-attention representations to dermoscopic patterns such as pigment networks, streaks, or blue-white veils.
* **Proposed Enhancement Plan:**
  1. **Two-Stage Adaptation:** Train linear head on cached features for 50 epochs (Stage 1), then unfreeze the **last 2 transformer blocks (Blocks 10 & 11)** of ViT-B/14 (Stage 2).
  2. **Hyperparameters:** Backbone $\text{lr} = 1 \times 10^{-5}$, Head $\text{lr} = 1 \times 10^{-4}$, weight decay $0.05$.
  3. **Regularization:** Early stopping with patience = 8 on `val_macro_f1`.

---

## 9. Controlled Experimental Sequence (Task 8)

All experiments will maintain the strict Revision V1 invariants:
- Same training manifest (`data/splits/revision_v1/train.csv`, 8,712 images)
- Same validation manifest (`data/splits/revision_v1/val.csv`, 1,875 images)
- Complete isolation of test manifest (`test.csv`, quarantined)
- Fixed random seed: 42
- Separate output directories and complete history tracking

| Experiment ID | Focus Area | Proposed Modification | Baseline for Comparison | Primary Evaluation Metrics |
| :---: | :--- | :--- | :--- | :--- |
| **A** | Preprocessing | Input scaling `mobilenet` ($[-1, 1]$) vs `div255` ($[0, 1]$) | Current Baseline (`div255`) | Accuracy, Macro F1, Val Loss |
| **B** | BatchNorm Handling | Freeze all 52 BN layers (`layer.trainable = False`) | Current Baseline (Trainable BN) | Accuracy, Macro F1, Training Stability |
| **C** | Fine-Tuning Strategy | Two-Stage: 10 epochs Head Warmup $\rightarrow$ 40 epochs Backbone at $\text{lr}=10^{-5}$ | Current Baseline (Full End-to-End) | Accuracy, Macro F1, Convergence Speed |
| **D** | Data Augmentation | Rotation ($\pm 45^\circ$), H/V Flips, Zoom ($\pm 10\%$) on Train only | Current Baseline (Zero Augmentation) | Generalization Accuracy, Macro F1 |
| **E** | Class Weighting | Balanced Inverse Class Weights ($w=[1.24, 0.92, 0.90]$) | Current Baseline (Unweighted) | Melanoma Recall, Macro F1 |
| **F** | DINOv2 Adaptation | Fine-tune top 2 ViT-B/14 blocks at $\text{lr}=10^{-5}$ with AdamW | Frozen DINOv2 Linear Probe | Macro F1, Accuracy, Mel Recall |
| **G** | Cumulative Synthesis | Best combined configuration (e.g. B + C + D + E) | Current Baseline | Overall Acc, Macro F1, All Class Recalls |

---

## 10. Risks and Limitations

1. **CPU Execution Speed:** Fine-tuning MobileNetV2 takes $\approx 15.5$ minutes per epoch on CPU ($\approx 12$ hours per 50-epoch experiment). Running multiple full 50-epoch experiments sequentially is time-intensive.
2. **Realistic Generalization Ceiling:** On a strictly patient-isolated, lesion-grouped split of ISIC 2019, standalone MobileNetV2 typically achieves **85% – 88%**. The objective is to achieve scientifically validated clinical generalization, rather than pursuing artificial metrics that rely on patient overlap.
3. **Data Integrity:** All model selection and stopping criteria must rely exclusively on `val.csv`. The `test.csv` partition will remain completely untouched until frozen evaluation.


# DermaVision — Final Consolidated Model Comparison Report

**Generated:** `2026-11-10 12:29:55 IST`  
**Workspace:** [Dermavision(ISIC2019 dataset)](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset))  
**Companion Structured Outputs:**
- Tabular Matrix (CSV): [final_model_comparison.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/final_model_comparison.csv)
- Machine-Readable Consolidated Bundle (JSON): [final_model_comparison.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/final_model_comparison.json)

**Experiment Integrity Statement:** Strictly **consolidated from existing verified artifacts** — zero models were retrained, re-quantized, pruned, modified, overwritten, or re-benchmarked. All metrics reflect identical evaluation over the legacy `2,489`-image ISIC 2019 validation split (`BCC = 665`, `Melanoma = 904`, `Nevus = 920`).

---

## 1. Executive Summary

Across all seven DermaVision model variants evaluated on the curated 2,489-image ISIC 2019 validation set, **MobileNetV2 FP32** delivers the highest classification accuracy (**90.9602%**), Macro F1 (**91.2665%**), and Melanoma recall (**87.0575%**), while **MobileNetV2 Full INT8 TFLite** achieves the strongest real-world edge-deployment trade-off (**90.7192%** accuracy, **91.0251%** Macro F1, **86.7257%** Melanoma recall, **2.94 MiB** disk size, and **4.51–16.35 ms** batch-1 latency depending on runtime conditions). Neither aggressive 50% magnitude pruning of MobileNetV2 (**-7.47 pp** accuracy, **-13.61 pp** Melanoma recall, zero dense CPU speedup) nor naive native PyTorch Dynamic INT8 quantization of DINOv2 ViT-B/14 (**-25.07 pp** accuracy, **-29.42 pp** Melanoma recall caused by Block 8–11 activation outliers) is safe to deploy clinically. Crucially, all results below measure performance on an **image-level split with 54.04% patient/lesion overlap leakage** against training, so they establish comparative compression and optimization trade-offs rather than independent clinical test generalization.

### Executive Decision Snapshot

| Model Variant | Val Acc (%) | Macro F1 (%) | Melanoma Recall (%) | Size on Disk (`MiB` = `1024^2` B) | Paired Pure `batch=1` Latency (Condition) | Deployment Verdict |
|---|---:|---:|---:|---:|---|---|
| **1. MobileNetV2 FP32 Original** | **90.9602%** | **91.2665%** | **87.0575%** | `30.21 MiB` (`31.67 MB`) | `104.89 ms` (`9.53 FPS`, Audit Run B `ONEDNN=1`) | **Best Classification Reference** |
| **2. MobileNetV2 Dynamic TFLite** | `90.8799%` (`90.88%`) | `91.1987%` (`91.20%`) | `85.6195%` (`85.62%`) | **`2.74 MiB`** (`2.88 MB`) | `27.76 ms` (`36.02 FPS`, Audit Run B `ONEDNN=1`) | Legacy App Fallback (`6.2x` slower than Full INT8) |
| **3. MobileNetV2 Full INT8 TFLite** | `90.7192%` (`90.72%`) | `91.0251%` (`91.03%`) | `86.7257%` (`86.73%`) | `2.94 MiB` (`3.08 MB`) | **`4.51 ms`** (`221.66 FPS`, Audit Run B `ONEDNN=1`) | **RECOMMENDED Practical Edge Winner** |
| **4. MobileNetV2 50% Pruned (Ep 5)** | `83.4873%` | `83.5474%` | `73.4513%` | `30.21 MiB` raw (`5.73 MiB` gz) | `101.54 ms` (`9.85 FPS`, Audit Run B `ONEDNN=1`) | **Do Not Replace Original** |
| **5. DINOv2 FP32 Original** | `86.6613%` | `86.8231%` | `79.0929%` | `330.34 MiB` (`346.39 MB`) | `373.35 ms` (`2.68 FPS`, Phase 2) / `235.09 ms` (Phase 1) | Server/Ensemble Research Backbone Only |
| **6. DINOv2 Native Dynamic INT8 (`b=1`)** `[1]` | `61.5910%` | `57.5372%` | `49.6681%` | `87.38 MiB` (`91.63 MB`) | `173.54 ms` (`5.76 FPS`, Phase 2 `1300 MHz`) | **Do Not Deploy** (Clinical Sensitivity Collapse) |
| **7. DINOv2 Folded-224 INT8 (Exp.)** `[2]` | `61.5910%` | `57.5372%` | `49.6681%` | `84.13 MiB` (`88.21 MB`) | `155.24 ms` (`6.44 FPS`, Phase 2 `1300 MHz`) | **Do Not Deploy** (Experimental Scratch Ablation) |

> **`[2]` Caveat for Model 7:** Full 2,489-image evaluation and pure CPU benchmarks exist in [dinov2_dynamic_int8_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_metrics.json) and [dinov2_dynamic_int8_report.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_report.md), while its serialized checkpoint (`88,211,872 B`) remains only in scratch (`dinov2_vitb14_folded_experimental_int8.pth`) and was not promoted to `models/dinov2/`. E2E latency was not benchmarked (`N/A`).

---

## 2. Comprehensive Classification Metrics Comparison (`n = 2,489` Validation Images)

All seven models were evaluated over the exact same `2,489` validation images (`BCC = 665`, `Melanoma = 904`, `Nevus = 920`; canonical index mapping `0 = BCC`, `1 = Melanoma`, `2 = Nevus`). Note: `MobileNetV2` models use `RGB 224x224` with pixel scaling `/ 255.0`; `DINOv2` models use `RGB 224x224` with ImageNet mean/std normalization.

### 2.1 Overall & Per-Class Recall Matrix

| # | Model | Correct / `2,489` | Val Accuracy | Macro Prec. | Macro Recall | Macro F1 | BCC Recall (`n=665`) | Melanoma Recall (`n=904`) | Nevus Recall (`n=920`) | Macro AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **MobileNetV2 FP32 Original** | **`2,264`** | **`90.9602%`** | **`91.1092%`** | **`91.4532%`** | **`91.2665%`** | **`96.5414%`** (`642`) | **`87.0575%`** (`787`) | `90.7609%` (`835`) | `0.9794` |
| **2** | **MobileNetV2 Dynamic TFLite** | `2,262` | `90.8799%` (`90.88%`) | `91.0943%` | `91.3586%` | `91.1987%` (`91.20%`) | `96.3910%` (`641`) | `85.6195%` (`774`) | `92.0652%` (`847`) | `N/A` |
| **3** | **MobileNetV2 Full INT8 TFLite** | `2,258` | `90.7192%` (`90.72%`) | `90.9928%` | `91.0672%` | `91.0251%` (`91.03%`) | `94.7368%` (`630`) | `86.7257%` (`784`) | `91.7391%` (`844`) | `N/A` |
| **4** | **MobileNetV2 50% Pruned (Ep 5)** | `2,078` | `83.4873%` | `84.8398%` | `83.3293%` | `83.5474%` | `82.4060%` (`548`) | `73.4513%` (`664`) | `94.1304%` (`866`) | `N/A` |
| **5** | **DINOv2 FP32 Original** | `2,157` | `86.6613%` | `86.5972%` | `87.2246%` | `86.8231%` | `93.2331%` (`620`) | `79.0929%` (`715`) | `89.3478%` (`822`) | `0.9680` |
| **6** | **DINOv2 Dynamic INT8 (`batch=1`)** `[1]` | `1,533` | `61.5910%` | `68.1669%` | `58.7073%` | `57.5372%` | `31.1278%` (`207`) | `49.6681%` (`449`) | **`95.3261%`** (`877`) | `0.8337` |
| **7** | **DINOv2 Folded-224 INT8 (`batch=1`)** `[2]` | `1,533` | `61.5910%` | `68.1669%` | `58.7073%` | `57.5372%` | `31.1278%` (`207`) | `49.6681%` (`449`) | **`95.3261%`** (`877`) | `0.8337` |

> **`[1]` Batch-Size Coupling Caveat:** Because `torch.ao.nn.quantized.dynamic.Linear` calculates one dynamic `(min, max)` activation scale across an entire input batch, evaluating `DINOv2 Native Dynamic INT8` at `batch=64` degrades further to **`60.5464%` accuracy** (`1,507/2,489`), **`56.2036%` Macro F1**, and **`48.1195%` Melanoma recall**, flipping `306` predictions (`12.29%`) compared to single-image `batch=1` inference. `batch=1` is therefore reported as the canonical deployment-aligned result.

### 2.2 Detailed Per-Class Precision & F1-Score Breakdown

| Model | BCC Prec. | BCC F1 | Melanoma Prec. | Melanoma F1 | Nevus Prec. | Nevus F1 | Deltas vs Own FP32 Baseline (`Acc` / `Macro F1` / `Mel Recall`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **MobileNetV2 FP32 Original** | `93.0435%` | `94.7601%` | `89.0271%` | `88.0313%` | `91.2568%` | `91.0082%` | `0.00 pp` / `0.00 pp` / `0.00 pp` |
| **MobileNetV2 Dynamic TFLite** | `93.5766%` | **`94.9630%`** | **`89.7912%`** | `87.6557%` | `89.9151%` | `90.9774%` | `-0.08 pp` / `-0.07 pp` / `-1.44 pp` |
| **MobileNetV2 Full INT8 TFLite** | **`94.0299%`** | `94.3820%` | `88.4876%` | `87.5978%` | `90.4609%` | **`91.0955%`** | `-0.24 pp` / `-0.24 pp` / **`-0.33 pp`** |
| **MobileNetV2 50% Pruned (Ep 5)** | `89.2508%` | `85.6919%` | `87.9470%` | `80.0482%` | `77.3214%` | `84.9020%` | `-7.47 pp` / `-7.72 pp` / **`-13.61 pp`** |
| **DINOv2 FP32 Original** | `86.3510%` | `89.6602%` | `85.5263%` | `82.1839%` | `87.9144%` | `88.6253%` | `0.00 pp` / `0.00 pp` / `0.00 pp` *(`-4.30 pp` Acc vs MNv2)* |
| **DINOv2 Dynamic INT8 (`b=1`)** | `83.1325%` | `45.2954%` | `64.6043%` | `56.1601%` | `56.7638%` | `71.1562%` | **`-25.07 pp`** / **`-29.29 pp`** / **`-29.42 pp`** |
| **DINOv2 Folded-224 INT8 (`b=1`)** | `83.1325%` | `45.2954%` | `64.6043%` | `56.1601%` | `56.7638%` | `71.1562%` | `-25.07 pp` / `-29.29 pp` / `-29.42 pp` *(`100%` identical to #6)* |

---

## 3. Model Footprint, RAM, Loading & CPU Latency Comparison

> [!IMPORTANT]
> **Critical Benchmark Condition Warning:** Latency measurements across the MobileNetV2 and DINOv2 reports were captured under **different TensorFlow `TF_ENABLE_ONEDNN_OPTS` flags (`"0"` vs `"1"`), different frameworks (TF/TFLite vs PyTorch), and varying Intel Core i5-1235U power/load states (`boost` vs `1300 MHz`)**. Never compare single numbers across isolated runs without reading their exact condition labels in Section 4. **Cold/warm model loading time** is cleanly separated below from **`batch=1` inference latency** and **end-to-end (from-disk JPEG) latency**.
>
> **Storage Unit Precision Note:** Existing MobileNetV2 reports divided byte counts by `1024^2` (binary **MiB**) while printing `"MB"` (`30.21 MB`, `2.74 MB`, `2.94 MB`), whereas DINOv2 reports labeled `1024^2` explicitly as **MiB** (`330.34 MiB`, `87.38 MiB`, `84.13 MiB`) alongside decimal `MB` (`10^6`). Both binary `MiB` and decimal `MB` are shown below for total clarity.

### 3.1 Storage Footprint, Model Loading, RAM Usage & End-to-End Latency

| # | Model | Exact Disk Bytes | Disk Size (`MiB` / `MB` dec.) | `gzip -9` Size | Model Load (Cold / Warm ms) | Incremental RAM (`Load` / `Total` Delta) | Peak Process RAM RSS | End-to-End Latency (From Disk JPEG) | E2E FPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **MobileNetV2 FP32 Original** | `31,673,490` | `30.21 MiB` (`31.67 MB`) | `26.92 MiB` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| **2** | **MobileNetV2 Dynamic TFLite** | `2,877,224` | **`2.74 MiB`** (`2.88 MB`) | `2.39 MiB` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| **3** | **MobileNetV2 Full INT8 TFLite** | `3,083,384` | **`2.94 MiB`** (`3.08 MB`) | `2.44 MiB` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| **4** | **MobileNetV2 50% Pruned (Ep 5)** | `31,673,383` | `30.21 MiB` (`31.67 MB`) | `5.73 MiB` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| **5** | **DINOv2 FP32 Original** | `346,390,400` | `330.34 MiB` (`346.39 MB`) | `N/A` | **P2:** `1581.1 / 1354.4 ms`<br>**P1:** `1113.8 / 1038.4 ms` | **P2:** `+662.32 / +994.13 MiB`<br>**P1:** `+666.92 / +749.75 MiB` | **P2:** `1,493.33 MiB`<br>**P1:** `1,113.16 MiB` | **P2:** `422.78 ± 28.86 ms`<br>**P1:** `254.78 ± 7.12 ms` | **P2:** `2.37`<br>**P1:** `3.92` |
| **6** | **DINOv2 Native Dynamic INT8** | `91,627,892` | **`87.38 MiB`** (`91.63 MB`) | `N/A` | **P2:** `2894.2 / 3168.7 ms` | **P2:** `+605.08 / +1084.39 MiB` | **P2:** `1,584.30 MiB` | **P2:** `176.61 ± 10.27 ms` | **P2:** `5.66` |
| **7** | **DINOv2 Folded-224 INT8 (Exp.)** | `88,211,872` | `84.13 MiB` (`88.21 MB`) | `N/A` | **P2:** `2800.8 / 2748.3 ms` | **P2:** `+609.47 / +1129.87 MiB` | **P2:** `1,628.85 MiB` | `N/A` *(Skipped in P2)* | `N/A` |

*(Legend: **P1** = DINOv2 Phase 1 Baseline Audit [`10:51`, uncontended sustained clock]; **P2** = DINOv2 Phase 2 Subprocess Benchmark [`12:12`, isolated subprocesses, `1300 MHz` reading]. In P2 E2E runs, JPEG open+resize+normalize consumed ~`19.8–19.9 ms` on average vs ~`12.25 ms` in P1.)*

### 3.2 Pure `batch=1` Inference Latency & Throughput Segregated by Benchmark Run

Every recorded benchmark below used **`batch_size = 1`**, **`50` warm-up + `200` timed iterations** on preloaded RAM tensors via `time.perf_counter()` (with model load & JPEG disk preprocessing **excluded**, except that Full INT8 MobileNetV2 **includes** numpy `int8` input quantization + output dequantization inside its timed loop):

#### A. MobileNetV2 Family (`TensorFlow 2.21.0` / `TFLite XNNPACK` on `12th Gen Intel Core i5-1235U`, Windows 11)

| Model | INT8 Report (`08:18`, `ONEDNN=0`, unlogged load) | Pruning Report (`09:29`, `ONEDNN=1`, unlogged load) | Audit Run A (`09:55`, `ONEDNN=0`, `37.8%` CPU) | Audit Run B (`09:57`, `ONEDNN=1`, `11.6%` CPU — **Controlled Paired**) |
|---|---|---|---|---|
| **MobileNetV2 FP32 Original** | `269.56 ± 83.04 ms` (`3.71 FPS`) | `131.77 ± 17.16 ms` (`7.59 FPS`) | `160.07 ± 25.10 ms` (`6.25 FPS`) | **`104.89 ± 9.95 ms`** (**`9.53 FPS`**) |
| **MobileNetV2 Dynamic TFLite** | `73.74 ± 9.22 ms` (`13.56 FPS`) | `38.17 ± 10.91 ms` (`26.20 FPS`) | `33.49 ± 4.89 ms` (`29.86 FPS`) | **`27.76 ± 1.77 ms`** (**`36.02 FPS`**) |
| **MobileNetV2 Full INT8 TFLite** | **`16.35 ± 2.43 ms`** (**`61.17 FPS`**) | `6.63 ± 2.29 ms` (`150.82 FPS`) | `6.85 ± 1.56 ms` (`146.08 FPS`) | **`4.51 ± 0.20 ms`** (**`221.66 FPS`**) |
| **MobileNetV2 50% Pruned (Ep 5)** | `N/A` *(Not yet created)* | `180.40 ± 34.64 ms` (`5.54 FPS`) | `154.46 ± 22.60 ms` (`6.47 FPS`) | **`101.54 ± 3.44 ms`** (**`9.85 FPS`**) |
| **Key Takeaways Within Run** | Full INT8 **`16.5x`** vs FP32 Keras | Full INT8 **`19.9x`** vs FP32; Pruned spiked | `ONEDNN=0` slows Keras FP32 | **Proves Pruned ≈ FP32 Keras (`101.5` vs `104.9 ms`); INT8 `23.3x` faster** |

#### B. DINOv2 ViT-B/14 Family (`PyTorch 2.10.0+cpu`, `onednn` / `MKL`, `threads=10` on `12th Gen Intel Core i5-1235U`)

| Model | Phase 1 Baseline (`10:51`, sustained boost) | Phase 2 Subprocess (`12:12`, `1300 MHz` — **Controlled Paired**) | Phase 2 Median (`p50`) & Range (`min` / `max`) | Phase 2 Pure Speedup vs FP32 |
|---|---:|---:|---|---|
| **DINOv2 FP32 Original** | `235.09 ± 5.93 ms` (**`4.25 FPS`**) | `373.35 ± 17.61 ms` (`2.68 FPS`) | `p50: 371.41 ms` (`347.51` / `508.79 ms`) | `1.00x` (Reference) |
| **DINOv2 Native Dynamic INT8** | `N/A` *(Phase 1 read-only)* | **`173.54 ± 27.55 ms`** (**`5.76 FPS`**) | `p50: 159.93 ms` (`149.38` / `309.86 ms`) | **`2.15x`** (`2.32x` by median) |
| **DINOv2 Folded-224 INT8 (Exp.)** | `N/A` *(Phase 2 ablation)* | `155.24 ± 9.88 ms` (`6.44 FPS`) | `p50: 152.58 ms` (`144.76` / `226.37 ms`) | `2.40x` |

---

## 4. Benchmark Methodology, Runtime Environment & Consistency Audit

### 4.1 Hardware, Software & Runtime Environment Specification

| Parameter | MobileNetV2 Family Experiments | DINOv2 ViT-B/14 Family Experiments |
|---|---|---|
| **Host Hardware** | `12th Gen Intel(R) Core(TM) i5-1235U` (`10` physical / `12` logical cores), `15.65 GB` RAM, CPU-only (no GPU) | Same host (`12th Gen Intel Core i5-1235U`, `10` physical / `12` logical cores, `15.65 GB` RAM, CPU-only) |
| **Operating System** | Windows 11 (`10.0.26300-SP0`, `AMD64`) | Windows 11 (`10.0.26300-SP0`, `AMD64`) |
| **Python & Deep Learning Stack** | `Python 3.13.5`, `TensorFlow 2.21.0`, `Keras 3.15.0` (runtime), `tfmot 0.8.1`, `NumPy`, `Pillow` | `Python 3.13.5`, `PyTorch 2.10.0+cpu`, `timm`/`torchvision`/`Pillow`, `psutil` |
| **CPU Execution Engine** | FP32 & Pruned: Keras dense Eigen CPU (`TF_ENABLE_ONEDNN_OPTS=0` vs `1`)<br>TFLite Dyn & Full INT8: `tf.lite.Interpreter` + XNNPACK Delegate | PyTorch `MKL` / `MKL-DNN` (`onednn` `qengine`, `torch.set_num_threads(10)`) |
| **Classification Eval Batch Size** | `batch=64` (FP32 Original, 50% Pruned); `batch=1` (Dynamic TFLite, Full INT8 TFLite) over all `2,489` images | `batch=64` (FP32 Original); **`batch=1` primary** (`batch=64` diagnostic) for Dynamic INT8 & Folded-224 over all `2,489` images |
| **Latency Benchmark Protocol** | `batch_size = 1`, `50` warm-up + `200` timed iterations (`time.perf_counter()`), preloaded tensors (load & JPEG decode excluded; Full INT8 loop includes numpy `int8` quant/dequant) | `batch_size = 1`, `50` warm-up + `200` timed pure preloaded iterations + separate `200` E2E JPEG-from-disk iterations + cold/warm load + `psutil` RSS memory (`Phase 2` ran in isolated fresh subprocesses at `1300 MHz`) |

### 4.2 Methodological & Cross-Run Consistency Notes

Comparing optimization outcomes cleanly requires accounting for four critical methodological factors verified across the project reports:

1. **TensorFlow `TF_ENABLE_ONEDNN_OPTS` & Host Load Variance (MobileNetV2):**
   - `quantize_and_evaluate_int8.py` ran at `08:18:17` with `TF_ENABLE_ONEDNN_OPTS="0"` and high unlogged host contention after TFLite conversion/evaluation (`269.56 ± 83.04 ms` FP32, `16.35 ms` INT8), whereas `evaluate_pruned_mobilenetv2.py` ran at `09:29:42` with `TF_ENABLE_ONEDNN_OPTS="1"` (`131.77 ms` FP32, `6.63 ms` INT8, and a noisy `180.40 ± 34.64 ms` spike on Pruned Keras).
   - Controlled back-to-back runs in [mobilenetv2_pruning_independent_audit.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_independent_audit.md) (`Audit Run A ONEDNN=0` vs `Audit Run B ONEDNN=1`) proved two facts: (a) enabling oneDNN cuts FP32 Keras latency by **`-34.5%`** (`160.07 -> 104.89 ms`), and (b) **unstructured 50% pruning yields virtually zero CPU speedup over dense FP32 Keras** (`101.54 vs 104.89 ms`) because standard Keras/Eigen BLAS executes dense matrix multiplication over zeroed elements identically (`0.0 * x = 0.0`).
2. **DINOv2 Phase 1 vs Phase 2 CPU Clock / Power State Difference:**
   - Phase 1 measured DINOv2 FP32 under higher sustained boost (`235.09 ms`, `4.25 FPS`), whereas Phase 2 ran all workers sequentially in isolated fresh Python subprocesses when `psutil.cpu_freq()` read `1300.0 MHz` (`373.35 ms`, `2.68 FPS`). Consequently, **Dynamic INT8 speedup (`2.15x` mean / `2.32x` median) must be anchored to Phase 2 paired FP32 (`373.35 ms -> 173.54 ms`)**, not Phase 1 FP32 (`235.09 ms`).
3. **Model Loading vs Inference vs End-to-End Scope Boundaries:**
   - **MobileNetV2 timings** cover inference on preloaded RAM tensors **excluding** model loading and JPEG disk decode/resize/`/255.0`. Only **Full INT8 TFLite** includes numpy `int8` input quantization and output dequantization inside its timed `4.51–16.35 ms` loop, meaning it already captures tensor-type adaptation overhead.
   - **DINOv2 timings** cleanly separate **cold loading** (`1,114–1,581 ms` FP32 vs `2,894 ms` Dynamic INT8 — slower because reloading invokes `quantize_dynamic` oneDNN packing before `load_state_dict()`), **pure `batch=1` tensor inference** (`173.54 ms` INT8 vs `373.35 ms` Phase 2 FP32), and **end-to-end from disk** (`176.61 ms` INT8 vs `422.78 ms` Phase 2 FP32, including ~`19.8–19.9 ms` `PIL` decode+preprocess).
4. **Cross-Family Architecture Contrast (No Apples-to-Oranges Confusion):**
   - Even comparing MobileNetV2's slowest heavily loaded INT8 measurement (`16.35 ms` at `ONEDNN=0`) against DINOv2 FP32's fastest uncontended Phase 1 measurement (`235.09 ms`), MobileNetV2 Full INT8 is **`14.4x` faster**; under paired clean/low-load conditions (`4.51 ms` vs `235.09 ms` Phase 1 or `173.54 ms` Phase 2 INT8), MobileNetV2 Full INT8 is **`52.1x` faster than DINOv2 FP32**, **`38.5x` faster than DINOv2 Dynamic INT8**, and **`112.3x` / `29.7x` smaller on disk**, while also delivering **`+4.06 pp`** higher validation accuracy (`90.72% vs 86.66%`) and **`+7.63 pp`** higher Melanoma recall (`86.73% vs 79.09%`) over DINOv2 FP32.

---

## 5. Class-Wise Analysis & Confusion Matrix Comparison

Inspection of the verified confusion matrices (`n = 2,489`; row = true class, column = predicted class `[BCC, Melanoma, Nevus]`) exposes a consistent, clinically dangerous failure mode whenever aggressive compression harms representation capacity:

| Model Variant | True `BCC` (`n=665`) `[B, M, N]` | True `Melanoma` (`n=904`) `[B, M, N]` | True `Nevus` (`n=920`) `[B, M, N]` | Melanomas Missed as `Nevus` (Worst Benign Error) |
|---|---|---|---|---|
| **MobileNetV2 FP32 Original** | `[642, 21, 2]` (`96.54%` recall) | `[39, 787, 78]` (`87.06%` recall) | `[9, 76, 835]` (`90.76%` recall) | **`78`** (`8.63%` of melanomas) |
| **MobileNetV2 Dynamic TFLite** | `[641, 22, 2]` (`96.39%` recall) | `[37, 774, 93]` (`85.62%` recall) | `[7, 66, 847]` (`92.07%` recall) | `93` (`10.29%` of melanomas) |
| **MobileNetV2 Full INT8 TFLite** | `[630, 32, 3]` (`94.74%` recall) | `[34, 784, 86]` (`86.73%` recall) | `[6, 70, 844]` (`91.74%` recall) | **`86`** (`9.51%` of melanomas) |
| **MobileNetV2 50% Pruned (Ep 5)** | `[548, 39, 78]` (`82.41%` recall) | `[64, 664, 176]` (**`73.45%` recall**) | `[2, 52, 866]` (`94.13%` recall) | **`176`** (**`19.47%`** — `2.26x` baseline!) |
| **DINOv2 FP32 Original** | `[620, 38, 7]` (`93.23%` recall) | `[83, 715, 106]` (`79.09%` recall) | `[15, 83, 822]` (`89.35%` recall) | `106` (`11.73%` of melanomas) |
| **DINOv2 Dynamic INT8 (`b=1`)** | `[207, 204, 254]` (**`31.13%` recall**) | `[41, 449, 414]` (**`49.67%` recall**) | `[1, 42, 877]` (`95.33%` recall) | **`414`** (**`45.80%`** of all melanomas!) |
| **DINOv2 Folded-224 INT8 (`b=1`)** | `[207, 204, 254]` (`31.13%` recall) | `[41, 449, 414]` (`49.67%` recall) | `[1, 42, 877]` (`95.33%` recall) | **`414`** (`45.80%` — identical logits) |

### Key Class-Wise Diagnostic Findings

1. **Both Malignant Classes Collapse Into Benign `Nevus` Under Failed Compression:**
   - **MobileNetV2 50% Pruned:** Pruning half the `Conv2D`/`Dense` weights over only 5 epochs (`Adam 1e-5`, zero constant-sparsity recovery epochs) shifted `176` true Melanomas (up from `78`) and `78` true BCCs (up from `2`) into benign `Nevus`, artificially boosting `Nevus` recall to `94.13%` while crushing `Nevus` precision from `91.26%` to `77.32%` and Melanoma recall by **`-13.61 pp`**.
   - **DINOv2 Dynamic INT8:** Collapsed massively onto `Nevus` (`1,545 / 2,489 = 62.07%` of all predictions vs `935` in FP32). More than **45.8% of true Melanomas (`414/904`)** and **38.2% of true BCCs (`254/665`)** were misdiagnosed as benign `Nevus`, dropping `BCC` recall by `-62.11 pp` and `Melanoma` recall by `-29.42 pp`. In clinical triage, predicting a malignant melanoma as a benign nevus carries the highest patient safety risk.
2. **Full INT8 MobileNetV2 Better Preserves Melanoma Sensitivity Than Dynamic-Range TFLite:**
   - While Dynamic-Range TFLite lost `-1.44 pp` Melanoma recall (`87.06% -> 85.62%`), Full INT8 TFLite trained-data activation calibration lost only **`-0.33 pp`** (`87.06% -> 86.73%`, only `3` fewer true Melanoma detections: `784` vs `787`), with a `-1.80 pp` shift in `BCC` recall (`96.54% -> 94.74%`). Overall Macro F1 stayed within **`-0.24 pp`** (`91.03%` vs `91.27%`).

---

## 6. Final Conclusions & Deployment Recommendations

1. **Best Overall Classification Performance — MobileNetV2 FP32 Original ([best_mobilenetv2.keras](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras)):**
   - Achieves the highest accuracy (**90.9602%**), Macro F1 (**91.2665%**), Macro AUC (**0.9794**), BCC recall (**96.5414%**), and Melanoma recall (**87.0575%**) among all individual evaluated models — outperforming DINOv2 FP32 by **`+4.30 pp`** accuracy, **`+4.44 pp`** Macro F1, and **`+7.96 pp`** Melanoma recall on this benchmark.
2. **Best Practical Edge-Deployment Trade-Off — MobileNetV2 Full INT8 TFLite ([best_mobilenetv2_full_int8.tflite](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite)):**
   - Delivers **90.7192%** accuracy (`-0.24 pp` vs FP32), **91.0251%** Macro F1 (`-0.24 pp`), and **86.7257%** Melanoma recall (`-0.33 pp`) while shrinking disk size by **90.27%** (`30.21 MiB -> 2.94 MiB`) and accelerating CPU `batch=1` inference to **`4.51 ms` (`221.66 FPS`)** under paired low-load XNNPACK execution (`16.35 ms` / `61.17 FPS` even under unoptimized `ONEDNN=0` contended runs).
3. **Why DINOv2 Native Dynamic INT8 Is Not Recommended Despite Smaller Size & Faster Inference:**
   - Although PyTorch `quantize_dynamic` reduced checkpoint size by **`-73.55%`** (`330.34 MiB -> 87.38 MiB`) and accelerated pure inference **`2.15x`** (`373.35 -> 173.54 ms`) and E2E inference **`2.39x`** (`422.78 -> 176.61 ms`), Phase 2 layer-wise diagnostics proved that register-free DINOv2 ViT-B/14 generates **extreme outlier tokens in Transformer Blocks 8–11** (peak `|x|` up to `494.72` vs `p99 ≈ 1.3–2.7`). Because PyTorch native dynamic INT8 uses a **single per-tensor `quint8 [0, 255]` scale** rather than per-token scaling, quantization step sizes of `~0.24–0.35` in `blocks.8.mlp.fc2` and `blocks.11.mlp.fc2` crush `[CLS]` token fidelity (cosine similarity collapses to **`0.1469`** at Block 11 output). Weight-quantization error itself was tiny (`< 0.0035`), so future DINOv2 optimization would require mixed-precision exclusion (`Blocks 8–11` kept FP32), weight-only INT8 (`W8A32`), or per-token activation quantization (e.g., ONNX Runtime/OpenVINO/torchao) — not naive `quantize_dynamic`.
4. **Why 50% Pruned MobileNetV2 Should Not Replace the Original:**
   - Five-epoch gradual pruning caused **`-7.47 pp` accuracy loss** (`83.49%` vs `90.96%`) and **`-13.61 pp` Melanoma recall drop** (`73.45%` vs `87.06%`). Moreover, standard dense `.keras` storage leaves file size unchanged (`30.21 MiB` uncompressed; savings appear only under `gzip -9` transfer at `5.73 MiB`), and standard TensorFlow/Keras CPU execution delivers **zero speedup** (`101.54 ms` vs `104.89 ms`) without a sparse execution kernel. Even compressed + gzipped, it is still ~`2.35x` larger than Full INT8 TFLite gzipped (`5.73 vs 2.44 MiB`) and **`-7.23 pp` lower in accuracy**.
5. **Why MobileNetV2 Full INT8 Is a Strong Deployment Candidate (and Frontend Alignment Note):**
   - Full integer quantization (`TFLITE_BUILTINS_INT8` with strict `int8` input/output, calibrated purely on `250` training images) eliminated all FP operator fallbacks, improved XNNPACK CPU throughput (`6.2x` faster than Dynamic-Range TFLite `27.76 ms -> 4.51 ms` in paired Audit Run B; `4.5x` faster in the `08:18` run `73.74 ms -> 16.35 ms`), preserved Melanoma recall at **`86.73%`** (`+1.11 pp` above Dynamic TFLite's `85.62%`), and enables native integer NPU/DSP execution on mobile/Raspberry Pi hardware. Standalone wrapper [prediction_int8.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py) already implements exact `int8` scale/zero-point pre/post-processing (`1/255.0, Z_in=-128`; `1/256.0, Z_out=-128`), while [streamlit_app.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py) / [prediction.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction.py) currently still loads `best_mobilenetv2.tflite` (Dynamic-Range).

---

## 7. Important Research Limitations & Unverified Claims Audit

> [!WARNING]
> **Critical Scientific Caveats — Must Accompany Any Publication or Clinical Discussion of These Numbers:**
>
> 1. **Lesion-ID Data Leakage in the `2,489`-Image Validation Set (`NO` Independent Test Set):**
>    [lesion_overlap_audit.txt](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/lesion_overlap_audit.txt) verified that the legacy `12,445`-image stratified 80:20 split (`random_state=42`) was performed strictly at the **image level** without grouping by `lesion_id`. Consequently, **`1,345 / 2,489` validation images (`54.04%`, spanning `964` unique lesions)** also appear in the `9,956`-image training set — including **`85.11%` of BCC validation images** (`566/665`) and **`79.31%` of Melanoma validation images** (`717/904`), while `1,886` curated images (`15.15%`, including `33.67%` of Nevus) have `NaN` `lesion_id`.
>    - On **clean non-overlapping lesions (`n = 1,144`)**, verified accuracy drops dramatically:
>      - **MobileNetV2 FP32 Original:** `94.50%` (overlapping `1,345`) vs **`86.80%`** (clean `1,144`) -> **`+7.70 pp` leakage inflation** (`Macro F1: 90.20% -> 81.31%`).
>      - **DINOv2 FP32 Original:** `88.92%` (overlapping `1,345`) vs **`84.00%`** (clean `1,144`) -> **`+4.92 pp` leakage inflation** (`Macro F1: 82.66% -> 75.66%`).
>    - Therefore, the `2,489`-image partition is a **leaky internal validation set**, **NOT** an independent test set, and these numbers **do not establish clinical generalization** to unseen patients or external clinical cohorts.
>
> 2. **Unverified & Conflicting Historical Ensemble / Diagram Claims (Excluded from Primary Comparison):**
>    - **`0.5/0.5` Ensemble (`91.36%`, [final_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/final_metrics.json)) & `0.4/0.6` Ensemble (`90.76%`, [ensemble_04_06_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_04_06_metrics.json)):** Both were evaluated on the exact same leaky `2,489` validation partition (dropping to `87.15%` and `86.80%` on non-overlapping lesions). Neither represents an independent test validation, and `0.4/0.6` trails standalone MobileNetV2 (`90.96%`).
>    - **Uncapped Split Mismatch (`72.35%`, [best_ensemble_classification_report.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/best_ensemble_classification_report.csv)):** [evaluate_final_ensemble.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py) evaluated `dinov2_classifier_best.pth` on an uncapped `4,144`-sample validation set (`2,575` Nevus samples), producing incompatible metrics.
>    - **Unsubstantiated HTML `"91.44%"` Test Accuracy Claim ([architecture_diagram.html](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram.html)):** Claims a `"70% Train / 15% Val / 15% Test"` split and `"91.44% Ensemble Test Acc"` (`+` `86.82%` DINOv2 CSV Epoch-35 curve for which no saved Ep35 checkpoint exists). **Unverified:** no held-out test script, split manifest, or test prediction artifact exists in the repository.

---

## 8. Source Files Used & Unverified / Unavailable Metrics (`N/A`)

### 8.1 Source Files Inventory (All Verified Unmodified via Pre/Post SHA-256)

- **MobileNetV2 Quantization & Pruning Artifacts:**
  - Reports & Metrics: [mobilenetv2_int8_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/int8/mobilenetv2_int8_metrics.json), [mobilenetv2_int8_verification_report.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/int8/mobilenetv2_int8_verification_report.md), [mobilenetv2_pruning_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_metrics.json), [mobilenetv2_pruning_report.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_report.md), [mobilenetv2_pruning_history.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_history.csv), [mobilenetv2_pruning_independent_audit.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_independent_audit.md)
  - Companion Audit Benchmark Raw JSONs (exact floats backing Audit Run A & Run B): [audit_part2_onednn_0.json](file:///C:/Users/HP/.gemini/antigravity/brain/2c749fb0-eed6-40fb-aabe-c32512150f72/scratch/audit_part2_onednn_0.json), [audit_part2_onednn_1.json](file:///C:/Users/HP/.gemini/antigravity/brain/2c749fb0-eed6-40fb-aabe-c32512150f72/scratch/audit_part2_onednn_1.json)
  - Optimization/Evaluation Scripts: [quantize_and_evaluate_int8.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/quantize_and_evaluate_int8.py), [prune_and_train_mobilenetv2.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prune_and_train_mobilenetv2.py), [evaluate_pruned_mobilenetv2.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_pruned_mobilenetv2.py)
  - Checkpoints verified (`bytes`): [best_mobilenetv2.keras](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) (`31,673,490`), [best_mobilenetv2.tflite](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.tflite) (`2,877,224`), [best_mobilenetv2_full_int8.tflite](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) (`3,083,384`), [best_mobilenetv2_pruned.keras](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_pruned.keras) (`31,673,383`)
- **DINOv2 Phase 1 & Phase 2 Artifacts:**
  - Reports & Metrics: [dinov2_baseline_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_baseline_metrics.json), [dinov2_deployment_baseline_audit.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_deployment_baseline_audit.md), [dinov2_layer_parameter_inventory.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_layer_parameter_inventory.json), [dinov2_dynamic_int8_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_metrics.json), [dinov2_dynamic_int8_report.md](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_report.md)
  - Inference Wrapper & Checkpoints verified (`bytes`): [dinov2_dynamic_int8_inference.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_dynamic_int8_inference.py), [dinov2_vitb14_pretrain.pth](file:///C:/Users/HP/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth) (`346,378,731`) + [dinov2_100_epoch_best.pth](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) (`11,669`), [dinov2_vitb14_dynamic_int8.pth](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_vitb14_dynamic_int8.pth) (`91,627,892`), experimental folded scratch file [dinov2_vitb14_folded_experimental_int8.pth](file:///C:/Users/HP/.gemini/antigravity/brain/2c749fb0-eed6-40fb-aabe-c32512150f72/scratch/dinov2_vitb14_folded_experimental_int8.pth) (`88,211,872`), secondary unverified 30-epoch head [dinov2_classifier_best.pth](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth) (`11,677`)
- **Shared Classification, Ensemble, Pipeline & Leakage Audit Artifacts:**
  - Metrics & History CSVs/JSONs: [final_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/final_metrics.json), [classification_report.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/classification_report.csv), [ensemble_04_06_metrics.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_04_06_metrics.json), [ensemble_classification_report.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_classification_report.csv), [ensemble_weight_comparison.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_weight_comparison.csv), [best_ensemble_classification_report.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/best_ensemble_classification_report.csv), [lesion_overlap_audit.txt](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/lesion_overlap_audit.txt), [dinov2_epoch_history.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2_epoch_history.csv), [mobilenetv2_epoch_history.csv](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/mobilenetv2_epoch_history.csv), [class_names.json](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/class_names.json)
  - Project App, Evaluation & Unrun Benchmark Scripts: [evaluation.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py), [evaluate_final_ensemble.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py), [benchmark.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/benchmark.py), [streamlit_app.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py), [prediction.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction.py), [prediction_int8.py](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py), [architecture_diagram.html](file:///C:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram.html)

### 8.2 Metrics Marked `N/A` / Could Not Be Verified

| Unavailable (`N/A`) or Unverified Metric | Affected Model(s) | Exact Reason / Audit Evidence |
|---|---|---|
| **Peak / Incremental Process RAM Usage (`MiB`)** | All 4 MobileNetV2 variants (`FP32`, `Dynamic TFLite`, `Full INT8`, `50% Pruned`) | Not instrumented or logged in `quantize_and_evaluate_int8.py`, `evaluate_pruned_mobilenetv2.py`, or the independent audit reports (`N/A`). *(Note: `benchmark.py` contains unused Raspberry Pi `psutil` code, with no saved run output.)* |
| **End-to-End Inference Latency from JPEG Disk** | All 4 MobileNetV2 variants + DINOv2 Folded-224 INT8 | MobileNetV2 benchmarks timed preloaded RAM tensors excluding disk JPEG decode/resize/`/255.0`. DINOv2 Phase 2 intentionally skipped E2E for Folded-224 (`end_to_end_from_disk_batch1: null` in JSON). |
| **Model Loading Time (Cold / Warm)** | All 4 MobileNetV2 variants | Not recorded in existing MobileNetV2 benchmark artifacts (`N/A`). |
| **Macro ROC-AUC & Weighted F1** | MobileNetV2 `Dynamic TFLite`, `Full INT8`, `50% Pruned` | `calc_metrics()` in INT8 and Pruning evaluation scripts did not compute/serialize `roc_auc_score` or `weighted_f1` (`N/A`). |
| **`gzip -9` Compressed Size** | All 3 DINOv2 variants | `gzip -9` archive compression was measured only in the MobileNetV2 pruning evaluation (`N/A` for DINOv2). |
| **Held-Out Independent Test Set Metrics & `"91.44%"` Ensemble Test Accuracy** | All models & ensembles | No isolated patient/lesion-grouped test set existed in historical experiments; HTML diagram `"70/15/15"` split & `"91.44%"` figure cannot be reproduced from any code or split manifest (`UNVERIFIED`). |

---

*Consolidation completed with zero model retraining, zero checkpoint edits, and zero overwriting of existing project reports.*

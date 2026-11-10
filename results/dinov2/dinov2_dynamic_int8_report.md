# DermaVision — DINOv2 Phase 2: Native PyTorch Dynamic INT8 Optimization Report

**Generated:** `2026-11-10T12:13:14`  
**Target Baseline:** DINOv2 ViT-B/14 (`dinov2_vitb14_pretrain.pth`) + [dinov2_100_epoch_best.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth)  
**Primary Artifact:** [dinov2_vitb14_dynamic_int8.pth](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_vitb14_dynamic_int8.pth)  
**Standalone Loader/Inference:** [dinov2_dynamic_int8_inference.py](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_dynamic_int8_inference.py)

---

## Executive Summary

| Evaluation Dimension | Original FP32 | Native Dynamic INT8 (`onednn`, `batch=1`) | Delta / Ratio | Status |
|---|---:|---:|---:|---|
| **Validation Accuracy (`n=2,489`)** | 86.6613% (`2157/2,489`) | 61.5910% (`1533/2,489`) | **-25.0703 pp** | `[LIMITATION]` Severe degradation |
| **Macro F1-Score** | 86.8231% | 57.5372% | **-29.2859 pp** | `[LIMITATION]` |
| **Melanoma Recall** | 79.0929% (`715/904`) | 49.6681% (`449/904`) | **-29.4248 pp** | `[LIMITATION]` Clinically unsafe |
| **Prediction Agreement vs FP32** | 100.00% (`2,489/2,489`) | 61.7115% (`1536/2,489`) | `953` flips (`38.29%`) | `[VERIFIED]` |
| **Quantized Param Coverage** | 0.00% (`0` quantized modules) | `49` Linear modules (`98.1949%` module params / `98.0991%` strict `qint8` weights) | `85,019,907` params in quantized Linear (`84,936,960` `qint8` weights + `82,947` `fp32` bias) | `[VERIFIED]` |
| **Serialized Size on Disk** | `330.34 MiB` (`346,390,400 B` Hub+head) / `330.31 MiB` (`346,399,209 B` single-file) | **`87.38 MiB` (`91,627,892 B`)** | **-73.55% (`3.78x` smaller)** | `[VERIFIED]` |
| **Pure Batch-1 Inference Latency (`50` warm / `200` timed)** | `373.35 ± 17.61 ms` (`2.68 FPS`) | **`173.54 ± 27.55 ms` (`5.76 FPS`)** | **`2.15x` faster (`-53.52%` latency)** | `[VERIFIED]` |
| **End-to-End from Disk (`200` JPEGs)** | `422.78 ± 28.86 ms` | **`176.61 ± 10.27 ms`** | **`2.39x` faster (`-58.23%` latency)** | `[VERIFIED]` |
| **Process RAM Delta on Load** | `+662.32 MiB` (`1161.52 MiB` RSS) | **`+605.08 MiB` (`1104.99 MiB` RSS)** | **`-57.24 MiB` load delta** | `[VERIFIED]` |

> **Deployment Verdict (`[LIMITATION]`):** Native PyTorch dynamic INT8 (`torch.ao.quantization.quantize_dynamic` with `onednn`) delivers genuine **3.78× disk size reduction** and **~2.15× CPU latency speedup**, but is **NOT clinically deployable** on this unmodified DINOv2 ViT-B/14 checkpoint due to a **-25.07 pp accuracy drop** and **-29.42 pp Melanoma recall drop** (`79.09% -> 49.67%`). Root-cause telemetry confirms weight quantization error is negligible (`< 0.0035`), whereas **high-norm artifact tokens in Blocks 8–11** (peak activation `|x|` up to `494.7` vs 99th percentile `~1.3–2.7`) saturate PyTorch's per-tensor dynamic `quint8` activation scaling, collapsing `[CLS]` representations toward Nevus predictions.

---

## 1. Environment, Backend & Original Model Gate Check

- **PyTorch & Backend:** `torch 2.10.0+cpu`, quantized engine `onednn` (supported: `['onednn']`), `MKL-DNN=True`, `OpenMP=True`, threads=`10`.
- **CPU & Memory:** `12th Gen Intel(R) Core(TM) i5-1235U`, `10` physical / `12` logical cores, `15.65 GiB` total RAM (`cpu_freq_current=1300.0 MHz` during benchmark).
- **Step 1 Gate Verification (`[VERIFIED]`):** Reconstructed FP32 model (`dinov2_vitb14` + `dinov2_100_epoch_best.pth`) evaluated on all `2,489` validation images reproduced Phase 1 exactly:
  - Accuracy: **`86.6613%`** (`2,157 / 2,489`)
  - Macro F1: **`86.8231%`**
  - Melanoma Recall: **`79.0929%`** (`715 / 904`)
  - Agreement with Phase 1 cached `dinov2_val_probabilities.npy`: **`2,489 / 2,489` (`100.00%`)**, max probability diff `4.74e-06`.

---

## 2. Native Dynamic INT8 Coverage & Unsupported Operations

Applied `torch.ao.quantization.quantize_dynamic(fp32_model, {nn.Linear}, dtype=torch.qint8)` with `torch.backends.quantized.engine = 'onednn'`. Notice PyTorch 2.10 emits a `DeprecationWarning` for `torch.ao.quantization` (pointing to external `torchao`, which is not installed per Phase 2 constraints).

### Quantized vs Non-Quantized Module Breakdown

| Layer Group | Module Type | Count | `qint8` Weight Elements | `fp32` Biases / Params | Total Params | Share of Total (`86,582,787`) |
|---|---|---:|---:|---:|---:|---:|
| `blocks.*.attn.qkv` (`768 -> 2304`) | `DynamicQuantizedLinear` | 12 | `21,233,664` | `27,648` | `21,261,312` | 24.56% |
| `blocks.*.attn.proj` (`768 -> 768`) | `DynamicQuantizedLinear` | 12 | `7,077,888` | `9,216` | `7,087,104` | 8.19% |
| `blocks.*.mlp.fc1` (`768 -> 3072`) | `DynamicQuantizedLinear` | 12 | `28,311,552` | `36,864` | `28,348,416` | 32.74% |
| `blocks.*.mlp.fc2` (`3072 -> 768`) | `DynamicQuantizedLinear` | 12 | `28,311,552` | `9,216` | `28,320,768` | 32.71% |
| `classifier` (`768 -> 3`) | `DynamicQuantizedLinear` | 1 | `2,304` | `3` | `2,307` | 0.003% |
| **Subtotal Quantized `Linear` Modules** | **`DynamicQuantizedLinear`** | **49** | **`84,936,960`** | **`82,947`** | **`85,019,907`** | **98.1949%** (`98.0991%` strict `qint8`) |
| `backbone` direct tensors (`pos_embed`, `cls_token`, `mask_token`) | `nn.Parameter` (`fp32`) | 3 tensors | — | `1,053,696` | `1,053,696` | 1.2170% |
| `patch_embed.proj` | `nn.Conv2d` (`fp32`) | 1 | — | `452,352` | `452,352` | 0.5225% |
| `blocks.*.norm1/2` + final `norm` | `nn.LayerNorm` (`fp32`) | 25 | — | `38,400` | `38,400` | 0.0444% |
| `blocks.*.ls1/2` | `LayerScale` (`fp32`) | 24 | — | `18,432` | `18,432` | 0.0213% |
| `GELU`, `Dropout`, `Identity`, `SDPA` | Functional / zero-param (`fp32`) | 62 | — | `0` | `0` | 0.00% |
| **Total Combined Model** | — | **161** | **`84,936,960`** | **`1,645,827`** | **`86,582,787`** | **100.0000%** |

- **Verification (`[VERIFIED]`):** All `49` `nn.Linear` layers pack weights into `torch.qint8` (`per_tensor_affine` reported inside `onednn` packed weights) with dynamic per-tensor `torch.quint8 [0, 255]` activation observers (`PlaceholderObserver(is_dynamic=True)`).
- **Self-Contained Reload Verification (`[VERIFIED]`):** Loading `models/dinov2/dinov2_vitb14_dynamic_int8.pth` offline without original pretrain/head files matches live quantized inference with `max_abs_diff = 0.00e+00`.

---

## 3. Classification Evaluation on All 2,489 Validation Images

Dynamic INT8 was evaluated primarily at **`batch=1`** (matching single-image deployment so each image receives its own dynamic activation scale), and additionally diagnosed at **`batch=64`**.

### Aggregate Metrics

| Metric | FP32 Baseline (`batch=64`) | Dynamic INT8 (`batch=1`, Primary) | Delta (`batch=1` vs FP32) | Dynamic INT8 (`batch=64`, Diagnostic) |
|---|---:|---:|---:|---:|
| **Accuracy** | 86.6613% (`2157/2,489`) | **61.5910%** (`1533/2,489`) | **-25.0703 pp** | 60.5464% (`1507/2,489`) |
| **Macro Precision** | 86.5972% | **68.1669%** | -18.4304 pp | 67.4449% |
| **Macro Recall** | 87.2246% | **58.7073%** | -28.5173 pp | 57.5896% |
| **Macro F1-Score** | 86.8231% | **57.5372%** | -29.2859 pp | 56.2036% |
| **Macro OvR ROC-AUC** | `0.9680` | **`0.8337`** | `-0.1343` | `0.8291` |

### Per-Class Performance (`batch=1` Dynamic INT8 vs FP32)

| Class (Support) | FP32 Prec | INT8 Prec | FP32 Recall | INT8 Recall | Recall Delta | FP32 F1 | INT8 F1 | F1 Delta | INT8 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BCC** (`665`) | 86.35% | 83.13% | 93.23% | **31.13%** | **-62.11 pp** | 89.66% | 45.30% | -44.36 pp | `0.8236` |
| **Melanoma** (`904`) | 85.53% | 64.60% | 79.09% | **49.67%** | **-29.42 pp** | 82.18% | 56.16% | -26.02 pp | `0.7718` |
| **Nevus** (`920`) | 87.91% | 56.76% | 89.35% | **95.33%** | **+5.98 pp** | 88.63% | 71.16% | -17.47 pp | `0.9057` |

### Confusion Matrices & Prediction Transitions (`n = 2,489`)

| True \ Predicted (`FP32` / `INT8 batch=1`) | Pred BCC | Pred Melanoma | Pred Nevus | Support |
|---|---:|---:|---:|---:|
| **True BCC** | `620` / **`207`** | `38` / **`204`** | `7` / **`254`** | 665 |
| **True Melanoma** | `83` / **`41`** | `715` / **`449`** | `106` / **`414`** | 904 |
| **True Nevus** | `15` / **`1`** | `83` / **`42`** | `822` / **`877`** | 920 |

- **Image-by-Image Agreement (`INT8 batch=1` vs `FP32`):**
  - Exact argmax agreement: **`1536 / 2,489` (`61.71%`)**
  - Predicted class changed (flips): **`953 / 2,489` (`38.29%`)**
  - Outcome shift: **`760` broken** (FP32 right -> INT8 wrong), `136` recovered (FP32 wrong -> INT8 right), `1397` both right, `139` both wrong (same), `57` both wrong (diff).
  - Attractor collapse toward Nevus: INT8 predicts **Nevus on `1,545 / 2,489` (`62.07%`)** images (vs FP32 `935`), including `414 / 904` (`45.80%`) true Melanomas (`374` FP32-predicted Melanomas flipped to Nevus).
- **FP32->INT8 Transition Matrix (rows=FP32 pred, cols=INT8 `batch=1` pred):** BCC->`[213, 228, 277]`, Melanoma->`[33, 429, 374]`, Nevus->`[3, 38, 894]`.
- **`[DISCREPANCY]` Cross-Batch Scale Coupling (`batch=1` vs `batch=64`):** Changing evaluation batch size alone from `1` to `64` shifts accuracy from `61.5910%` to `60.5464%` and Melanoma recall from `49.6681%` to `48.1195%`, flipping **`306 / 2,489` (`12.29%`)** predictions because `DynamicQuantizedLinear` computes one per-tensor `(min, max)` across all `B` images in the batch.

---

## 4. Root-Cause Diagnosis of Accuracy Degradation (Outlier Activation Tokens)

Measured exclusively on **training images** (zero validation data leakage) to isolate whether errors arise from weight quantization or dynamic activation scaling:

- **Weight quantization error is negligible:** `max|W_fp32 - W_dequant_int8|` across inspected Linear layers is `0.00063–0.00347`. Feeding FP32 backbone features into the INT8 quantized `classifier` yields an output difference of just `0.074` (`isolated CLS cos = 0.99993`).
- **High-norm artifact tokens appear cleanly at Block 8:** Across 10 sampled training images, Blocks 0–7 show bounded activations (`max|x| ≈ 1.1–7.7`, token norm max/median `1.46–3.32×`, `[CLS]` cosine similarity **`0.9950–0.9987`**). Starting at Block 8, register-free `dinov2_vitb14` concentrates extreme magnitudes onto background artifact tokens:

| Transformer Block | Mean `FP32 max|x|` | Max `FP32 max|x|` | Mean `FP32 p99|x|` | Mean `max / p99` Ratio | Token Norm Max/Median | Mean `[CLS]` CosSim (FP32 vs INT8) |
|---:|---:|---:|---:|---:|---:|---:|
| Block  0 | `1.50` | `1.57` | `0.159` | `9.5x` | `1.46x` | **`0.9976`** |
| Block  1 | `1.12` | `1.14` | `0.126` | `8.9x` | `1.80x` | **`0.9961`** |
| Block  2 | `1.64` | `1.66` | `0.131` | `12.5x` | `2.09x` | **`0.9982`** |
| Block  3 | `1.77` | `1.80` | `0.167` | `10.6x` | `2.14x` | **`0.9979`** |
| Block  4 | `3.84` | `3.89` | `0.201` | `19.2x` | `2.78x` | **`0.9987`** |
| Block  5 | `2.31` | `2.42` | `0.276` | `8.4x` | `1.49x` | **`0.9950`** |
| Block  6 | `4.98` | `5.14` | `0.432` | `11.5x` | `2.49x` | **`0.9957`** |
| Block  7 | `7.71` | `7.91` | `0.610` | `12.7x` | `3.32x` | **`0.9953`** |
| Block  8 | `396.67` | `431.30` | `1.312` | `308.6x` | `55.05x` | **`0.8619`** |
| Block  9 | `428.20` | `463.26` | `1.768` | `244.7x` | `36.39x` | **`0.8814`** |
| Block 10 | `459.48` | `494.72` | `2.666` | `172.7x` | `22.76x` | **`0.7682`** |
| Block 11 | `158.11` | `189.00` | `5.037` | `31.5x` | `3.68x` | **`0.1469`** |

- **Isolated Sublayer Culprit:** Feeding the **exact identical FP32 input tensor** from `train[0]` (`ISIC_0014225_downsampled`) directly into each quantized Linear module in isolation shows where dynamic activation quantization breaks:

| Isolated Layer (Exact Same FP32 Input) | Weight Dequant MaxDiff | Input `max|x|` | Input `p99|x|` | Isolated `[CLS]` CosSim | Isolated `[CLS]` MaxDiff |
|---|---:|---:|---:|---:|---:|
| `backbone.blocks.8.attn.qkv` | `0.00063` | `26.152` | `4.126` | **`0.99879`** | `0.1597` |
| `backbone.blocks.8.attn.proj` | `0.00100` | `3.091` | `0.832` | **`0.99886`** | `0.0186` |
| `backbone.blocks.8.mlp.fc1` | `0.00172` | `41.745` | `4.809` | **`0.99948`** | `0.3474` |
| `backbone.blocks.8.mlp.fc2` | `0.00107` | `88.690` | `0.614` | **`0.57746`** | `1.0625` |
| `backbone.blocks.10.mlp.fc1` | `0.00064` | `41.406` | `5.374` | **`0.99980`** | `0.2314` |
| `backbone.blocks.10.mlp.fc2` | `0.00166` | `4.753` | `0.337` | **`0.99450`** | `0.0893` |
| `backbone.blocks.11.mlp.fc1` | `0.00347` | `19.429` | `4.209` | **`0.99842`** | `0.2929` |
| `backbone.blocks.11.mlp.fc2` | `0.00144` | `61.911` | `0.553` | **`0.11906`** | `4.8082` |
| `classifier` | `0.00117` | `6.250` | `4.677` | **`0.99993`** | `0.0742` |

Because `torch.ao.quantization.quantize_dynamic` assigns a **single global `quint8 [0, 255]` scale per tensor call** (not per-token), the post-GELU outlier in `blocks.8.mlp.fc2` (`max = 88.69` vs `p99 = 0.614`, step size `~0.35`) and `blocks.11.mlp.fc2` (`max = 61.91` vs `p99 = 0.553`, step size `~0.24`) crushes normal `[CLS]` and patch token representations into just a few integer bins, destroying `[CLS]` semantics (`0.1469` mean cosine similarity by Block 11 output).

---

## 5. Step 3 Experimental Ablation: Fixed 224×224 `pos_embed` Folding (Isolated)

Tested separately in memory without altering `dinov2_vitb14_pretrain.pth` or `models/dinov2/dinov2_vitb14_dynamic_int8.pth`:

- **FP32 Folded (`[1, 1370, 768] -> [1, 257, 768]`) Equivalence across all 2,489 Val Images (`[VERIFIED]`):** `2489/2,489` (`100.00%`) agreement, `max_abs_prob_diff = 0.00e+00`, `max_abs_logit_diff = 0.00e+00`.
- **INT8 Folded vs Primary INT8 (`batch=1`, `2,489` Val Images):** `2489/2,489` (`100.00%`) agreement, `max_abs_prob_diff = 0.00e+00` (`61.5910%` acc, `49.6681%` Melanoma recall).
- **Size & Latency Impact of Folding Alone:** Reduces checkpoint size by `-3.26 MiB` (`91,627,892 B -> 88,211,872 B`, coverage `99.174%`), while pure `batch=1` latency changes trivially (`373.35 ms -> 334.79 ms` FP32; `173.54 ms -> 155.24 ms` INT8). Because runtime interpolation overhead is negligible (<1–2 ms) and preserving `[1, 1370, 768]` retains original model flexibility, keeping unmodified `pos_embed` in `models/dinov2/dinov2_vitb14_dynamic_int8.pth` is appropriate.

---

## 6. Model Size, Memory, Loading & CPU Latency Benchmarks

Each benchmark worker ran in a fresh isolated Python subprocess under identical CPU hardware/PyTorch settings (`50` warm-up + `200` timed `batch=1` iterations on preloaded RAM tensors; JPEG disk loading/preprocessing timed separately).

### Model Footprint & Loading Time

| Metric | Original FP32 (Hub + Head) | Original FP32 (Single File) | Primary Dynamic INT8 (`models/dinov2/...`) | Exp. Folded INT8 (Scratch) |
|---|---:|---:|---:|---:|
| **Serialized Size on Disk** | `330.34 MiB (346.39 MB)` (`346,390,400 B`) | `330.35 MiB (346.40 MB)` | **`87.38 MiB (91.63 MB)` (`91,627,892 B`)** | `84.13 MiB (88.21 MB)` |
| **Cold Load Time (1st)** | `1581.13 ms` | `1806.20 ms` | **`2894.19 ms`** | `2800.80 ms` |
| **Warm Reload Mean (3 trials)** | `1354.38 ms` | `1622.56 ms` | **`3168.72 ms`** | `2748.33 ms` |
| **Process RSS Before Load** | `499.20 MiB` | `497.87 MiB` | `499.91 MiB` | `498.98 MiB` |
| **Process RSS After Load** | `1161.52 MiB` (`+662.32 MiB`) | `1156.10 MiB` (`+658.23 MiB`) | **`1104.99 MiB` (`+605.08 MiB`)** | `1108.45 MiB` (`+609.47 MiB`) |
| **Peak RSS After Inference** | `1493.33 MiB` | `1472.22 MiB` | **`1584.30 MiB`** | `1628.85 MiB` |

### Pure `batch=1` Inference & Separate End-to-End CPU Latency

| Benchmark Variant | Mean ± Std (ms) | Median p50 (ms) | p95 / Max (ms) | Throughput (FPS) | Speedup vs FP32 |
|---|---:|---:|---:|---:|---:|
| **Original FP32 (Pure `batch=1`)** | `373.35 ± 17.61` | `371.41` | `398.51 / 508.79` | `2.68` | `1.00x` |
| **Dynamic INT8 Preserved (Pure `batch=1`)** | **`173.54 ± 27.55`** | **`159.93`** | `237.58 / 309.86` | **`5.76`** | **`2.15x`** |
| *Exp. Folded Fixed-224 FP32 (Pure)* | `334.79 ± 9.77` | `332.25` | `356.19 / 378.42` | `2.99` | `1.12x` |
| *Exp. Folded Fixed-224 INT8 (Pure)* | `155.24 ± 9.88` | `152.58` | `172.64 / 226.37` | `6.44` | `2.40x` |
| **Original FP32 (End-to-End from Disk)** | `422.78 ± 28.86` | `417.67` | `472.97 / 651.61` | `2.37` | `1.00x` |
| **Dynamic INT8 Preserved (End-to-End from Disk)** | **`176.61 ± 10.27`** | **`176.76`** | `194.64 / 227.91` | **`5.66`** | **`2.39x`** |

> **Notes on Benchmark Dynamics:**
> - **Throughput & Median vs Mean:** Preprocessing (`Image.open` + `Resize(224, 224)` + `Normalize`) takes ~`19.79 ms` in INT8 E2E vs ~`19.94 ms` in FP32 E2E. Both FP32 and INT8 workers ran back-to-back under identical system power/thermal state (`1300 MHz` reported by `psutil`, slower than Phase 1's `235.09 ms` under higher sustained boost). Comparing median (`p50`) latency (`159.93 ms` vs `371.41 ms`), pure INT8 speedup is **`2.32x`** (`2.15x` by mean, `2.39x` E2E).
> - **Loading Overhead & Runtime RSS:** Reloading INT8 takes longer (`2894 ms` vs `1581 ms`) because `load_dynamic_int8_model()` instantiates an FP32 ViT-B/14 graph and runs `quantize_dynamic()` `onednn` packing before `load_state_dict()`. Post-load RSS drops `-57.24 MiB` (`605.08 MiB` vs `662.32 MiB`), while `onednn` runtime scratchpad buffers during inference bring active RSS delta to `+1084.39 MiB` vs `+994.13 MiB`.

---

## 7. Preservation Compliance & Recommendations

- **Preservation Check (`[VERIFIED]`):** All `18` monitored original files and all `3` Phase 1 deliverable reports match their pre-Phase-2 SHA-256 checksums identically (**`0` mutated files**).
- **Why Do NOT Deploy Current Full-Linear Dynamic INT8 as Clinical Replacement:** Although disk size drops `-73.55%` and inference speeds up cleanly, losing `-29.42 pp` Melanoma recall (`45.8%` of Melanoma cases predicted Nevus) violates clinical safety.
- **Actionable Path for Phase 3 (Without Mixing Untested Dependencies Here):**
  1. **Selective / Mixed-Precision Quantization (`Blocks 8–11` kept FP32, `Blocks 0–7` Dynamic INT8):** Since Blocks 0–7 maintain `> 0.995` `[CLS]` cosine similarity under dynamic INT8 (`diagnose` telemetry), selectively excluding `blocks.8`–`11` (or `blocks.*.mlp.fc2` layers with outlier post-GELU activations) preserves majority size/speed gains while potentially eliminating the Block 8–11 activation quantization collapse.
  2. **Per-Token Dynamic Activation Quantization / Activation Smoothing:** Per-token `qint8`/`quint8` quantization (e.g., `torchao` when permitted, or `ONNX Runtime` with `MatMulNBits`/per-token/QDQ smoothing + SmoothQuant scale migration) isolates outlier tokens from corrupting the `[CLS]` token's quantization scale.
  3. **Weight-Only INT8 (W8A32) or Register-Trained DINOv2 (`dinov2_vitb14_reg`):** Weight-only INT8 achieves nearly identical `~73%` storage savings with zero activation quantization error (`max weight err < 0.0035`), avoiding outlier-token collapse altogether.

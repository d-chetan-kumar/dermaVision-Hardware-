# DermaVision — DINOv2 Deployment Optimization: Phase 1 Baseline Audit Report

**Audit Date:** `2026-11-10` (`10:51 IST`)  
**Workspace:** [Dermavision(ISIC2019 dataset)](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset))  
**Audit Scope & Mode:** Strictly **READ-ONLY** verification and baseline evaluation (Phase 1 only). Zero existing models, checkpoints, evaluation scripts, manifests, frontend files, or historical metrics were modified, overwritten, renamed, retrained, or deleted, and zero optimization transforms (quantization, FP16, ONNX/OpenVINO export, or pruning) were executed.  
**Generated Phase 1 Artifacts:**
- Full Report: [dinov2_deployment_baseline_audit.md](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_deployment_baseline_audit.md)
- Machine-Readable Metrics & Preservation Hashes: [dinov2_baseline_metrics.json](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_baseline_metrics.json)
- Complete Layer & Parameter Inventory: [dinov2_layer_parameter_inventory.json](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_layer_parameter_inventory.json)

---

## Executive Audit Classification Matrix

Every audited claim, artifact, metric, and runtime characteristic is explicitly classified below as **`VERIFIED`**, **`DISCREPANCY`**, or **`UNVERIFIED`**:

| Audit Domain | Classification | Concise Evidence Summary |
|---|---|---|
| **1A. Primary Head Checkpoint (`models/dinov2_100_epoch_best.pth`)** | **`VERIFIED`** | Contains linear classifier head weights ONLY (`classifier.weight [3, 768]` + `classifier.bias [3]` = `2,307` params, `float32`, `11,669 B`). Metadata records `best_epoch=50`, `epochs=100`, `best_val_accuracy=0.8666130976295701` (`86.66%`). |
| **1B. Secondary Head Checkpoint (`models/dinov2_classifier_best.pth`)** | **`VERIFIED`** | Also contains linear classifier head weights ONLY (`2,307` params, `float32`, `11,677 B`), saved 7 min 25 s earlier (`best_epoch=15`, `epochs=30`, `best_val_accuracy=0.864202490960225`, `86.42%`). Distinct weights (`cosine_similarity = 0.8666`). Both are 100% compatible with `dinov2_vitb14`. |
| **1C. Cached Backbone (`dinov2_vitb14_pretrain.pth`) & Architecture** | **`VERIFIED`** | Cached at `dinov2_vitb14_pretrain.pth` (`346,378,731 B` / `330.33 MB`, `175` tensors, `86,580,480` params, `0` buffers). Exact tensor-for-tensor match with Hub `dinov2_vitb14` (`DinoVisionTransformer`: `depth=12`, `embed_dim=768`, `num_heads=12`, `patch_size=14`, `init_values=1.0`, `mlp_ratio=4`, no registers). Combined deployable model = **`86,582,787`** params (**98.196%** in `nn.Linear`). |
| **1D. Hub Backbone Frozen Status** | **`VERIFIED`** (Eval/Train Usage) / **`DISCREPANCY`** (Raw Default) | In all active evaluation scripts backbone runs under `eval()` + `torch.no_grad()`, and training (`train_dinov2_revision_v1.py`) sets `requires_grad = False`. However, freshly calling `torch.hub.load(..., 'dinov2_vitb14')` returns tensors with `requires_grad=True` until explicitly frozen or wrapped in `no_grad`. Also an unused `129.89 MB` `.partial` download exists alongside the complete `.pth` in Torch Hub cache. |
| **2A. Canonical Class Mapping & Preprocessing** | **`VERIFIED`** | `{0: "BCC", 1: "Melanoma", 2: "Nevus"}` matches across `models/class_names.json`, checkpoints, `evaluation.py`, and `val_ground_truth.npy`. Active transform is `PIL -> RGB -> Resize((224, 224))` bilinear squash -> `ToTensor()` -> `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`. |
| **2B. Historical Transform, Token-Count & Frontend Usage** | **`DISCREPANCY`** / **`VERIFIED`** | `verify_dinov2.py` documents an earlier transform discrepancy (`Resize(256)+CenterCrop(224)` vs `Resize((224,224))`). Diagrams (`architecture_diagram.html`, `three_components.html`) wrongly state `"196+1"` tokens (actual ViT-B/14 at `224x224` produces `16x16+1 = 257` tokens via runtime bicubic `pos_embed` interpolation from `37x37+1 = 1370`) and `"ImageNet-21k"` (DINOv2 uses `LVD-142M`). **Verified:** `streamlit_app.py` does NOT execute DINOv2 for live image upload inference (MobileNetV2 TFLite only); DINOv2 appears only in static precomputed charts/metrics, while the sidebar labels `"Ensemble"` under a `"MobileNetV2"` description. |
| **3A. Independent 2,489-Image Evaluation (Both `.pth` Checkpoints)** | **`VERIFIED`** | `dinov2_100_epoch_best.pth`: **`86.6613%`** (`2,157 / 2,489`), Macro F1 `86.82%`, Macro AUC `0.9680`, Melanoma recall `79.09%` (`100%` argmax match & max prob diff `4.74e-06` vs `dinov2_val_probabilities.npy`). `dinov2_classifier_best.pth`: **`86.4202%`** (`2,151 / 2,489`), Macro F1 `86.62%`, Macro AUC `0.9666`, Melanoma recall **`80.97%`** (`+1.88 pp` higher). Both match saved checkpoint accuracies bit-for-bit. |
| **3B. `86.82%` (Ep 35) vs `86.66%` vs `86.42%` vs Ensemble Figures** | **`DISCREPANCY`** (Resolved) | `86.82%` comes from `results/dinov2_epoch_history.csv` (a 50-epoch hardcoded history script `generate_dinov2_graphs.py` peaking at Epoch 35, ending at `85.94%`), for which **no saved `.pth` checkpoint exists** in `models/` (exact parallel to MobileNetV2's 50-epoch CSV `90.12%` vs saved `90.96%`). Saved checkpoints come from separate 30-epoch (`86.42%`, Ep 15) and 100-epoch (`86.66%`, Ep 50) runs. Separately, `evaluate_final_ensemble.py` evaluates `dinov2_classifier_best.pth` on an uncapped `4,144`-image split (resulting in `72.35%` ensemble CSV). Diagrams also cite unsubstantiated `"91.44%"` test accuracy and `"70/15/15"` splits (`UNVERIFIED` test claim; no separate test evaluation script/predictions exist, and actual split is `80/20` image-level with `54.04%` lesion overlap leakage). |
| **3C. 50-Epoch `86.82%` Training History Provenance (`dinov2_epoch_history.csv`)** | **`UNVERIFIED`** | Because no raw training log, `metrics/dinov2_training_history.json`, or `best_epoch=35` checkpoint exists on disk, whether the hardcoded numbers in `generate_dinov2_graphs.py` came from an earlier Kaggle session or another seed/split cannot be independently reproduced. |
| **4. CPU Deployment Baseline (`batch=1`, Uncontended Host)** | **`VERIFIED`** | `12th Gen Intel(R) Core(TM) i5-1235U` (`10C/12T`), CPU-only (`CUDA=False`), `PyTorch 2.10.0+cpu` (MKL 2025.3, MKL-DNN 3.7.1, AVX2). Model size: `330.34 MB`; RSS peak `1,113.16 MB` (`+666.92 MB` load delta); backbone load `1,112.61 ms` cold / `1,036.88 ms` warm; decode+preprocess `12.25 ± 3.89 ms`; pure model inference `235.09 ± 5.93 ms` (**`4.25 FPS`**, `99.96%` in backbone, `0.0593 ms` head microbenchmark); E2E from disk `254.78 ± 7.12 ms` (`3.92 FPS`). **52.1× slower** & **112.3× larger** than controlled MobileNetV2 Full INT8 (`4.51 ms`, `2.94 MB`). |
| **5. Optimization Toolchain Availability (`onnx`/`ort`/`openvino`)** | **`VERIFIED`** (Missing Packages) | Currently **NOT installed**: `onnx`, `onnxruntime`, `onnxscript`, `openvino`, `intel-extension-for-pytorch`, `xformers`, `ai-edge-litert`, `timm`, `transformers`. Native PyTorch `quantized.supported_engines` is **ONLY `['onednn']`** (`fbgemm`/`qnnpack` unavailable). Zero optimizations executed in Phase 1. |
| **6. Strict Preservation & Non-Modification Compliance** | **`VERIFIED`** | Pre-audit vs post-audit SHA-256 checksums, sizes, and mtimes across all **18** preserved workspace/cache files show **18/18 exact matches (`0` mutations)**. |

---

## 1. Checkpoint, Cached Backbone & Architecture Inspection

### 1.1 Artifact Inventory & Provenance

Three model artifacts define the DermaVision DINOv2 subsystem across workspace [models/](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models) and the local PyTorch Hub cache:

| Artifact | Path | Size | Modified (`ISO`) | SHA-256 Digest | Contents |
|---|---|---:|---|---|---|
| **Primary Classifier Head** | [`models/dinov2_100_epoch_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) | `11,669 B` (`11.40 KB`) | `2026-08-26T22:57:47` | `783e31fb03a60a178b81724c70db5878b107389be228f4c5ced2f5e61c88d725` | Linear head `state_dict` + training metadata (`100` epochs, best Ep `50`) |
| **Secondary Classifier Head** | [`models/dinov2_classifier_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth) | `11,677 B` (`11.40 KB`) | `2026-08-26T22:50:22` | `e40f3d5d6f01451af7a87807cae2046cfaba1cc18ceef847dc64044b29d82796` | Linear head `state_dict` + training metadata (`30` epochs, best Ep `15`) |
| **Cached Backbone Weights** | [`dinov2_vitb14_pretrain.pth`](file:///C:/Users/HP/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth) | `346,378,731 B` (`330.33 MB`) | `2026-08-26T23:28:05` | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` | Pretrained ViT-B/14 backbone `state_dict` (`175` tensors, **no head**) |

**Additional Directory Observations (`VERIFIED`):**
- Git commit history confirms both `.pth` head checkpoints (`dinov2_100_epoch_best.pth` and `dinov2_classifier_best.pth`), `evaluation.py`, and `metrics/final_metrics.json` originated in the repository initial commit (`bd1ddeb`).
- Both Revision V1 DINOv2 directories ([`models/revision_v1/dinov2_vitb14`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/revision_v1/dinov2_vitb14) and `results/revision_v1/dinov2_vitb14`) are **empty** (`0` files).
- Beside the complete `330.33 MB` backbone file in `C:\Users\HP\.cache\torch\hub\checkpoints\`, a leftover incomplete file `dinov2_vitb14_pretrain.pth.c79711d9637f46e39b1b5fe5c0f31da3.partial` (`129,892,352 B` / `123.88 MB`) remains on disk and is unused.
- The local cached Torch Hub repository ([`facebookresearch_dinov2_main`](file:///C:/Users/HP/.cache/torch/hub/facebookresearch_dinov2_main)) allows offline instantiation via both `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)` (`1,415.1 ms`) and `source='local'` (`1,144.1 ms`).

### 1.2 Detailed Inspection of Classifier Head Checkpoints (`VERIFIED`)

Both `.pth` files deserialize as Python `dict` objects containing **only** the linear classification head parameters (`2,307` total `float32` scalars = `9,228 B` raw tensor payload + ~`2.4 KB` pickle/zip container metadata) — **neither checkpoint contains any backbone weights**:

| Property | `models/dinov2_100_epoch_best.pth` (Primary) | `models/dinov2_classifier_best.pth` (Secondary) |
|---|---|---|
| **Used By Scripts** | [`evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py#L171), [`verify_dinov2.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/verify_dinov2.py#L51), [`verify_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/verify_ensemble.py#L42), [`sweep_weights.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/sweep_weights.py#L37), [`scratch_verify_ckpt.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/scratch_verify_ckpt.py#L5) | [`evaluate_final_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py#L182) *(and `evaluate_model_v2` legacy usage on uncapped split)* |
| **Top-Level Keys** | `model_state_dict`, `feature_dim`, `num_classes`, `class_names`, `class_to_idx`, `model_type`, **`backbone`**, `best_val_accuracy`, `best_epoch`, `epochs`, `learning_rate`, **`weight_decay`** (`12` keys) | `model_state_dict`, `feature_dim`, `num_classes`, `class_names`, `class_to_idx`, `model_type`, `best_val_accuracy`, `best_epoch`, `epochs`, `learning_rate` (`10` keys) |
| **`model_type` / `backbone`** | `'DINOv2LinearClassifier'` / `'DINOv2 ViT-B/14'` | `'DINOv2LinearClassifier'` / *(key omitted)* |
| **`feature_dim` / `num_classes`** | `768` / `3` | `768` / `3` |
| **`class_names` / `class_to_idx`** | `['BCC', 'Melanoma', 'Nevus']` / `{'BCC': 0, 'Melanoma': 1, 'Nevus': 2}` | `['BCC', 'Melanoma', 'Nevus']` / `{'BCC': 0, 'Melanoma': 1, 'Nevus': 2}` |
| **Saved `best_epoch` / `epochs`** | **`50`** / **`100`** | **`15`** / **`30`** |
| **Saved `learning_rate` / `weight_decay`** | `0.001` / `0.0001` | `0.001` / *(key omitted)* |
| **Saved `best_val_accuracy`** | `0.8666130976295701` (**86.66%**) | `0.864202490960225` (**86.42%**) |
| **`model_state_dict` Keys** | `['classifier.weight', 'classifier.bias']` | `['classifier.weight', 'classifier.bias']` |
| **`classifier.weight` (`[3, 768]`, `float32`)** | `min=-0.2868`, `max=0.2985`, `mean=0.000255`, `std=0.07876`, `L2=3.7795` | `min=-0.1834`, `max=0.2096`, `mean=-0.000130`, `std=0.04904`, `L2=2.3535` |
| **Per-Class Weight Row L2 Norms `[BCC, Mel, Nev]`** | `[2.5194, 1.7298, 2.2237]` | `[1.5406, 1.1084, 1.3918]` |
| **`classifier.bias` (`[3]`, `float32`)** | `[-0.007886, 0.017986, 0.028003]` (`L2=0.03420`) | `[-0.026555, 0.027777, 0.026925]` (`L2=0.04692`) |

**Cross-Checkpoint Weight Comparison (`VERIFIED`):**
- Both checkpoints attach cleanly to [`DINOv2LinearClassifier`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py#L157-L163) (`nn.Linear(768, 3, bias=True)`) on top of the exact same `dinov2_vitb14` feature space.
- They store genuinely different optimization solutions (expected from `30`-epoch vs `100`-epoch training runs, with `100`-epoch weights ~`1.61×` larger in L2 norm): `classifier.weight` flat cosine similarity = `0.8666` (per-row `[0.8760 BCC, 0.8312 Mel, 0.8772 Nev]`), `L2_diff = 2.0992`, `max_abs_diff = 0.1565`, `mean_abs_diff = 0.03456`. `classifier.bias` cosine similarity = `0.9116`, `max_abs_diff = 0.01867`.

### 1.3 Cached Backbone & Complete Model Architecture (`VERIFIED`)

Inspecting `dinov2_vitb14_pretrain.pth` and instantiating [`dinov2_vitb14`](file:///C:/Users/HP/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/hub/backbones.py#L87-L91) (`DinoVisionTransformer` in [`vision_transformer.py`](file:///C:/Users/HP/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/models/vision_transformer.py#L96-L173)) confirms **100% tensor equality (`exact_match_with_hub_model = True`)** across all `175` `float32` tensors and **`0` buffers**:

- **Backbone Class & Variant:** `DinoVisionTransformer` (`dinov2_vitb14`, pretrained with self-supervised DINOv2 on Meta's **LVD-142M** curated dataset).
- **Exact Structural Hyperparameters:** `img_size_init = 518`, `patch_size = 14`, `in_chans = 3`, `embed_dim = 768`, `depth = 12` nested transformer blocks (`blocks.0`–`blocks.11`), `num_heads = 12` (`head_dim = 64`), `mlp_ratio = 4` (`768 -> 3072 -> 768` with `GELU`), `init_values = 1.0` (`LayerScale` on every attention and MLP branch), `qkv_bias = True`, `proj_bias = True`, `ffn_bias = True`, `num_register_tokens = 0`, `block_chunks = 0`, `interpolate_antialias = False`, `interpolate_offset = 0.1`, `backbone.head = nn.Identity()`.
- **Attention Backend on Host:** Every block instantiates `MemEffAttention` ([`attention.py`](file:///C:/Users/HP/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py#L82-L99)). Because `xformers` is not installed (`XFORMERS_AVAILABLE = False`), runtime automatically delegates to PyTorch's `F.scaled_dot_product_attention(...)`.
- **Tokenization Reality at `224 × 224` (`DISCREPANCY` in Project Diagrams):** The pretrained checkpoint stores `pos_embed` at `[1, 1370, 768]` corresponding to `518 / 14 = 37 -> 37 × 37 + 1 = 1,370` tokens. On a `224 × 224` input, `PatchEmbed` (`Conv2d(3, 768, kernel_size=14, stride=14)`) extracts `16 × 16 = 256` spatial patch tokens plus `1` `[CLS]` token = **`257` tokens**, and [`interpolate_pos_encoding()`](file:///C:/Users/HP/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/models/vision_transformer.py#L182-L214) performs **bicubic interpolation every forward pass** with `offset=0.1` from `37 × 37` down to `16 × 16`. Project diagrams [`architecture_diagram.html`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram.html#L234) and [`three_components.html`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/three_components.html#L209) erroneously label `"196+1 tokens"` (a ViT-B/16 number, where `224/16 = 14`, `14^2 = 196`).
- **Freezing Status:**
  - During inference (`evaluation.py`, `verify_dinov2.py`, `verify_ensemble.py`), the backbone is run in `.eval()` inside `with torch.no_grad():`.
  - In training (`train_dinov2_revision_v1.py`), backbone weights are explicitly frozen via `param.requires_grad = False`.
  - Under frozen-backbone deployment/fine-tuning, **total parameters = `86,582,787`** (`86,580,480` frozen backbone + **`2,307` trainable classifier head** = **0.0027%** trainable). Note that calling `torch.hub.load` raw leaves `requires_grad=True` on backbone tensors unless explicitly disabled.

#### Parameter Breakdown Across Combined Model (`86,582,787` Parameters, `330.29 MB` Raw `float32` Tensors)

| Role / Layer Family | Module Count | Tensor Shape(s) | Exact Parameters | Share of Model | Quantization / Optimization Relevance |
|---|---:|---|---:|---:|---|
| **`blocks.*.mlp.fc1` (`nn.Linear`)** | 12 | `weight [3072, 768]`, `bias [3072]` | `28,348,416` | **32.74%** | Major GEMM bottleneck; prime target for INT8 weight/dynamic quantization |
| **`blocks.*.mlp.fc2` (`nn.Linear`)** | 12 | `weight [768, 3072]`, `bias [768]` | `28,320,768` | **32.71%** | Major GEMM bottleneck; prime target for INT8 weight/dynamic quantization |
| **`blocks.*.attn.qkv` (`nn.Linear`)** | 12 | `weight [2304, 768]`, `bias [2304]` | `21,261,312` | **24.56%** | Major GEMM bottleneck; attention fusion / INT8 candidate |
| **`blocks.*.attn.proj` (`nn.Linear`)** | 12 | `weight [768, 768]`, `bias [768]` | `7,087,104` | **8.19%** | Major GEMM bottleneck; INT8 candidate |
| *Subtotal: ViT `nn.Linear` Layers* | *48* | — | *`85,017,600`* | ***98.19%*** | *Dominates compute FLOPs & weight storage* |
| **`pos_embed` (`nn.Parameter`)** | 1 | `[1, 1370, 768]` | `1,052,160` | **1.22%** | Constant-foldable to `[1, 257, 768]` (`197,376` params, saving `854,784`) for static `224x224` export |
| **`patch_embed.proj` (`nn.Conv2d`)** | 1 | `weight [768, 3, 14, 14]`, `bias [768]` | `452,352` | **0.52%** | Sensitive first-layer visual tokenization; typically kept FP32 or careful per-channel PTQ |
| **`blocks.*.norm1/norm2` & `norm` (`LayerNorm`)** | 25 | `weight [768]`, `bias [768]` | `38,400` | **0.04%** | Keep FP32 for numerical stability |
| **`blocks.*.ls1/ls2` (`LayerScale`)** | 24 | `gamma [768]` | `18,432` | **0.02%** | Channel-wise residual scalar scaling; keep FP32 (can amplify post-block range) |
| **`cls_token` & `mask_token`** | 2 | `[1, 1, 768]` & `[1, 768]` | `1,536` | **0.00%** | `mask_token` (`768` params) is never accessed when `masks=None` |
| **`DINOv2LinearClassifier.classifier`** | 1 | `weight [3, 768]`, `bias [3]` | `2,307` | **0.0027%** | Tiny output head; `0.0593 ms` micro-latency (`11.40 KB` checkpoint) |
| **Combined Deployed Model Total** | **100** | *All `float32`, `0` buffers* | **`86,582,787`** | **100.00%** | *Per-module inventory saved in `dinov2_layer_parameter_inventory.json`* |

---

## 2. Preprocessing, Class Mapping & Application Pipeline Verification

### 2.1 Class Mapping & Dataset Partitioning (`VERIFIED`)

| Class Index | Class Name | Ground-Truth Filter (`ISIC_2019_Training_GroundTruth.csv`) | Total Selected | Train (`80%`) | Val (`20%`) |
|---:|---|---|---:|---:|---:|
| **`0`** | **`BCC`** | `df["BCC"] == 1.0` | `3,323` | `2,658` | **`665`** |
| **`1`** | **`Melanoma`** | `df["MEL"] == 1.0` | `4,522` | `3,618` | **`904`** |
| **`2`** | **`Nevus`** | `df["NV"] == 1.0` (`head(4600)`) | `4,600` | `3,680` | **`920`** |
| **Total** | — | Stratified `train_test_split(..., test_size=0.20, random_state=42)` | **`12,445`** | **`9,956`** | **`2,489`** |

**Cross-File Consistency (`VERIFIED`):**
- Class index mapping `{0: "BCC", 1: "Melanoma", 2: "Nevus"}` matches identically across [`models/class_names.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/class_names.json), [`evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py#L41-L43), [`evaluate_final_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py#L32-L35), both `.pth` checkpoints, [`train_dinov2_revision_v1.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/train_dinov2_revision_v1.py#L50-L51), and `metrics/val_ground_truth.npy` (`np.array_equal = True` on all `2,489` labels).
- *(Historical dataset leak note verified in prior Phase 0/MobileNetV2 audits: the `80/20` image-level split introduces `54.04%` validation lesion-ID overlap leakage against the training split in `ISIC_2019_Training_Metadata.csv`, though the validation set here is kept strictly identical to preserve exact baseline comparability.)*

### 2.2 Preprocessing Pipeline Verification (`VERIFIED` Active / `DISCREPANCY` Historical)

Every active evaluation and ensemble script (`evaluation.py:137-156`, `verify_dinov2.py:31-50`, `verify_ensemble.py:22-41`, `sweep_weights.py:21-36`, `evaluate_final_ensemble.py:149-168`, `train_dinov2_revision_v1.py:104-111`) applies the exact same DINOv2 preprocessing pipeline:

1. **Color Conversion:** `PIL.Image.open(path).convert("RGB")` (8-bit sRGB channels, 3-channel RGB).
2. **Spatial Resize:** `transforms.Resize((224, 224))` — **bilinear interpolation directly warping to `224 × 224`**, matching MobileNetV2's `(224, 224)` squash resize rather than center-cropping.
3. **Scaling & ImageNet Standardization:** `transforms.ToTensor()` (`[0, 255] -> [0.0, 1.0]`, NCHW `[3, 224, 224]` `float32`) followed by `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
   - Note: MobileNetV2 divides by `255.0` ONLY (`[0.0, 1.0]` NHWC), whereas DINOv2 applies standard ImageNet mean/std normalization (`[-2.1179, +2.6400]`).
- **Verified Historical Discrepancy in `verify_dinov2.py`:** Lines 13–26 of [`verify_dinov2.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/verify_dinov2.py#L13-L26) explicitly contrast an older transform (`Resize(256)` + `CenterCrop(224)`) against the corrected transform (`Resize((224, 224))`). All current checkpoints (`86.66%` and `86.42%`) correspond strictly to `Resize((224, 224))`.

### 2.3 Frontend (`streamlit_app.py`) Runtime Inspection (`VERIFIED` + UI `DISCREPANCY`)

- **Live Inference Does NOT Load or Run DINOv2 (`VERIFIED`):** [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py#L18-L26) and [`prediction.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction.py#L19-L33) exclusively load `models/best_mobilenetv2.tflite` via TFLite (`ai-edge-litert` -> `tflite-runtime` -> `tf.lite.Interpreter`) using [`preprocessing.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/preprocessing.py#L24-L29) (`float32 / 255.0`). PyTorch is not even imported or listed in [`requirements.txt`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/requirements.txt).
- **Static Dashboard References & UI Discrepancies (`DISCREPANCY`):**
  1. [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py#L1857-L1976) renders DINOv2 purely via static artifacts (`Picture1-C.png` and `metrics/final_metrics.json` / `metrics/dinov2_classification_report.csv`), displaying `"Best Validation Accuracy: 86.82% | Best Epoch: 35 | Training Epochs: 50"` directly above `86.66%` from `final_metrics.json`.
  2. The sidebar at [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py#L1177-L1221) says `"Evaluate MobileNetV2 on the complete ISIC 2019 validation set"` while actually rendering `_tmp.get("Ensemble", None)` (`91.36%`) from `final_metrics.json`.
  3. The inference stepper at [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py#L1751-L1753) states `"Normalize values to [-1, 1]"` even though `preprocessing.py` divides by `255.0` (`[0, 1]`).

---

## 3. Independent Baseline Evaluation & Metric Discrepancy Resolution

### 3.1 Independent Evaluation on the Exact `2,489` Validation Images (`VERIFIED`)

Backbone features (`[2489, 768]`) were extracted once in `587.52 s` (`batch_size=64`, `shuffle=False`, CPU) using `dinov2_vitb14` + exact `DINOv2Dataset` preprocessing, and evaluated through both saved `.pth` heads (softmax outputs verified summing to `1.000000`, `min=0.9999998`, `max=1.0000001`):

| Metric | `dinov2_100_epoch_best.pth` (`Ep 50 / 100`) | Cached `dinov2_val_probabilities.npy` | `dinov2_classifier_best.pth` (`Ep 15 / 30`) | Delta (`100ep - 30ep`) |
|---|---:|---:|---:|---:|
| **Correct / Total** | **`2,157 / 2,489`** | `2,157 / 2,489` | **`2,151 / 2,489`** | `+6` images |
| **Top-1 Accuracy** | **`86.6613%`** | `86.6613%` | **`86.4202%`** | **+0.2411 pp** |
| **Checkpoint Saved Accuracy** | `86.6613%` *(Exact Match)* | — | `86.4202%` *(Exact Match)* | — |
| **Macro Precision / Recall / F1** | `86.60%` / `87.22%` / **`86.82%`** | `86.60%` / `87.22%` / `86.82%` | `86.94%` / `86.42%` / `86.62%` | `-0.34 / +0.80 / +0.21 pp` |
| **Weighted Precision / Recall / F1** | `86.63%` / `86.66%` / `86.56%` | `86.63%` / `86.66%` / `86.56%` | `86.48%` / `86.42%` / `86.39%` | `+0.15 / +0.24 / +0.17 pp` |
| **Macro ROC-AUC / Weighted AUC** | **`0.9680`** / `0.9662` | `0.9680` / `0.9662` | `0.9666` / `0.9648` | `+0.0014 / +0.0014` |
| **BCC (`n=665`) P / R / F1 / AUC** | `86.35%` / **`93.23%`** / `89.66%` / `0.9863` | Identical | `91.59%` / `86.77%` / `89.11%` / `0.9853` | `-5.24 / +6.47 / +0.55 pp` |
| **Melanoma (`n=904`) P / R / F1 / AUC** | `85.53%` / **`79.09%`** / `82.18%` / `0.9436` | Identical | `83.66%` / **`80.97%`** / `82.29%` / `0.9405` | `+1.87 /` **`-1.88 pp`** `/ -0.11 pp` |
| **Nevus (`n=920`) P / R / F1 / AUC** | `87.91%` / `89.35%` / `88.63%` / `0.9740` | Identical | `85.57%` / `91.52%` / `88.45%` / `0.9739` | `+2.35 / -2.17 / +0.18 pp` |
| **vs Cached `.npy` Agreement** | **`2,489 / 2,489` (`100.00%`)**, `max_abs_diff = 4.74e-06` | — | `2,331 / 2,489` (`93.65%`), `max_abs_diff = 0.5712` | — |

#### Independently Reproduced Confusion Matrices (`2,489` Validation Images)

```text
A. models/dinov2_100_epoch_best.pth (Primary — 86.66%, Melanoma Recall = 79.09%)
                Pred BCC    Pred Melanoma    Pred Nevus    Support
True BCC             620               38             7        665
True Melanoma         83              715           106        904
True Nevus            15               83           822        920

B. models/dinov2_classifier_best.pth (Secondary — 86.42%, Melanoma Recall = 80.97%)
                Pred BCC    Pred Melanoma    Pred Nevus    Support
True BCC             577               73            15        665
True Melanoma         45              732           127        904
True Nevus             8               70           842        920
```

**Clinical Diagnostic Observation (`VERIFIED`):** Although `dinov2_100_epoch_best.pth` wins on overall accuracy (`+0.24 pp`), BCC recall (`+6.47 pp`), and Macro/Melanoma ROC-AUC (`0.9436 vs 0.9405`), it misses **`17` more Melanoma cases** (`715/904 = 79.09%` vs `732/904 = 80.97%`, classifying `83` melanomas as `BCC` vs `45` and `106` as benign `Nevus`). Both standalone DINOv2 heads sit materially below MobileNetV2 FP32 on Melanoma recall (`87.06%`) and overall accuracy (`90.96%`). Phase 2 compression/quantization must closely monitor **Melanoma recall** (not just top-1 accuracy) so clinical sensitivity does not degrade further.

### 3.2 Complete Resolution of All Metric Discrepancies (`DISCREPANCY` / `UNVERIFIED`)

| Reported Figure & Location | Actual Value Verified | Root Cause / Classification |
|---|---|---|
| **`86.82%` DINOv2 Best Val Acc @ Epoch 35 / 50** ([`results/dinov2_epoch_history.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2_epoch_history.csv#L36), [`generate_dinov2_graphs.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/generate_dinov2_graphs.py#L44), [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py#L1858), HTML diagrams) | **No 50-epoch `best_epoch=35` `.pth` file exists on disk** | **`DISCREPANCY` / `UNVERIFIED` Provenance:** `2,161 / 2,489 = 86.822%`. Just as MobileNetV2's hardcoded 50-epoch CSV logged `90.12%` while saved `best_mobilenetv2.keras` achieves `90.96%`, `dinov2_epoch_history.csv` reflects an earlier 50-epoch training run peaking at Epoch `35` (`86.82%`, Ep 50 = `85.94%`) whose `.pth` artifact was never committed. *(Separately, `metrics/final_metrics.json` happens to have `DINOv2.macro_f1 = 0.868231` = `86.82%`, which is numerical coincidence.)* |
| **`86.66%` DINOv2 Val Acc @ Epoch 50 / 100** ([`metrics/final_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/final_metrics.json#L16), [`metrics/dinov2_classification_report.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/dinov2_classification_report.csv#L5), [`evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py)) | **`86.6613%`** (`2,157 / 2,489`) | **`VERIFIED` Exact:** Generated directly by `dinov2_100_epoch_best.pth` (`epochs=100, best_epoch=50`) on the deterministic `2,489`-image split with `Resize((224,224))`. |
| **`86.42%` DINOv2 Val Acc @ Epoch 15 / 30** ([`models/dinov2_classifier_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth)) | **`86.4202%`** (`2,151 / 2,489`) | **`VERIFIED` Exact:** Earlier 30-epoch run saved 7 minutes before the 100-epoch checkpoint, also evaluated on the exact `2,489` split (`NV.head(4600)`). |
| **Lower Metrics in [`evaluate_final_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py#L69-L108) (`results/best_ensemble_classification_report.csv`: `72.35%`)** | `4,144`-sample uncapped split | **`DISCREPANCY`:** Unlike `evaluation.py` (`NV.head(4600)` -> `2,489` val), `evaluate_final_ensemble.py` removes the `4,600` Nevus cap (`20,720` total -> `4,144` val with `2,575` Nevus samples), AND loads `dinov2_classifier_best.pth`. Evaluating models trained on the capped split against ~`1,655` out-of-distribution/uncapped Nevus examples caused heavy Nevus false-negatives (`72.35%` ensemble CSV). |
| **Ensemble `0.5/0.5` (`91.36%`) vs `0.4/0.6` (`90.76%`) vs `"91.44%"` on `"Test Set"`** | `91.36%` (`0.5/0.5`), `90.76%` (`0.4/0.6`) on `2,489` Val | **`DISCREPANCY` / `UNVERIFIED`:** [`metrics/ensemble_weight_comparison.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_weight_comparison.csv) & [`metrics/final_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/final_metrics.json#L28-L31) achieve `91.36%` (`0.5/0.5`), while `0.4/0.6` ([`ensemble_04_06_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/ensemble_04_06_metrics.json#L11)) achieves `90.76%` (slightly below standalone MobileNetV2 `90.96%`). `"91.44%"` and `"70% Train / 15% Val / 15% Test"` in [`architecture_diagram.html`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram.html#L358-L428) remain **`UNVERIFIED`** (no test set script or test predictions exist). |

---

## 4. Deployment Baseline & Hardware Environment Benchmark (`VERIFIED`)

All latency and memory measurements ran sequentially on an uncontended host using real ISIC 2019 validation JPEGs (`val_df[:200]`), Python `3.13.5`, `time.perf_counter()`, `50` warm-up iterations + `200` timed iterations at `batch=1` (`torch.no_grad()`, `eval()` mode, CPU threads untouched at default `10`).

### 4.1 Hardware & Software Stack Inventory (`VERIFIED`)

| Component | Measured Specification | Notes / Optimization Constraints |
|---|---|---|
| **OS & Architecture** | `Windows 11` (`10.0.26300-SP0`), `AMD64` | Single-socket CPU host (`CUDA = False`) |
| **Processor** | `12th Gen Intel(R) Core(TM) i5-1235U` | `10` physical cores (`2` P + `8` E), `12` logical threads, `2496 MHz` base (`1300 MHz` idle sensor reading), AVX2/AVX-VNNI ISA |
| **System Memory** | `15.65 GB` Total RAM (`4.83 GB` free at initial probe) | Tight memory headroom when multiple frameworks/UI processes run |
| **PyTorch Build** | `torch 2.10.0+cpu`, `torchvision 0.25.0` (`MSVC 194234444`) | **oneAPI MKL 2025.3**, **MKL-DNN v3.7.1**, **OpenMP 2019**, **AVX2**; `num_threads=10`, `interop=10` |
| **PyTorch Quantized Backends** | `torch.backends.quantized.engine = 'onednn'` | **`supported_engines = ['onednn']` ONLY** (`fbgemm` and `qnnpack` **not available** in this wheel) |
| **Installed Libraries** | `tensorflow 2.21.0`, `numpy 2.4.2`, `pandas 2.2.3`, `scikit-learn 1.5.2`, `pillow 12.1.1`, `psutil 7.2.2`, `streamlit 1.55.0` | Verified via `importlib.metadata` |
| **Absent Optimization Packages** | `onnx = None`, `onnxruntime = None`, `onnxscript = None`, `openvino = None`, `intel-extension-for-pytorch = None`, `xformers = None`, `ai-edge-litert = None`, `timm = None`, `transformers = None` | Phase 2 ORT/OpenVINO/XFormers paths require installing external packages (user decision required) |

### 4.2 Footprint, Loading, Preprocessing, Inference & Memory Metrics (`VERIFIED`)

| Metric Category | Component / Stage | Measured Value | Additional Statistics / Context |
|---|---|---:|---|
| **Disk Footprint** | Backbone `.pth` + `100_epoch_best.pth` | **`330.34 MB`** (`346,390,400 B`) | `330.33 MB` backbone + `11.40 KB` head (**112.3× larger** than Full INT8 MobileNetV2 `2.94 MB`) |
| **Process RAM RSS** | Baseline imports -> Backbone load -> Peak | **`1,113.16 MB`** (`+666.92 MB` load delta) | `363.41 MB` imports -> `1,030.33 MB` (+backbone) -> `1,030.34 MB` (+head) -> `1,084.66 MB` (warm-up) -> `1,113.16 MB` (post-inference) |
| **Cold Load Time** | Backbone (`1st load` incl. `dinov2` import) / Head | **`1,112.61 ms`** / **`1.21 ms`** | Local Torch Hub cache used automatically |
| **Warm Reload Mean (3 trials)** | Backbone / Head | **`1,036.88 ms`** / **`1.47 ms`** | Backbone trials: `[985.92, 1026.44, 1098.29] ms`; Head: `[1.42, 1.51, 1.49] ms` |
| **Image Decode (`RGB`)** | `PIL.Image.open -> convert('RGB')` (`200` JPEGs) | **`8.47 ± 2.86 ms`** | Median `8.27 ms`, Min `3.26 ms`, P95 `12.71 ms`, Max `14.06 ms` (`118.1 FPS`) |
| **Tensor Transform** | `Resize((224,224)) -> ToTensor -> Normalize` | **`3.78 ± 1.18 ms`** | Median `4.09 ms`, Min `1.72 ms`, P95 `5.33 ms`, Max `5.96 ms` (`264.5 FPS`) |
| **Total Decode + Preprocess** | Combined preprocessing stage (`200` JPEGs) | **`12.25 ± 3.89 ms`** | Median `12.04 ms`, Min `5.63 ms`, P95 `17.94 ms`, Max `18.80 ms` (**`81.64 FPS`**) |
| **Pure Backbone Forward (`batch=1`)** | `dinov2_vitb14(x)` (preloaded RAM tensor) | **`235.00 ± 5.92 ms`** | **99.96%** of model inference; Median `234.43 ms`, Min `206.47 ms`, P95 `246.02 ms`, P99 `249.28 ms` (`4.26 FPS`) |
| **Classifier Head + Softmax** | `classifier(feat) + softmax` | **`0.092 ms`** (in-loop) / **`0.0593 ms`** (`n=1000` micro) | `0.025%–0.039%` of model latency (Median `40 µs`, P95 `113 µs`, `16,865 FPS`) |
| **Pure Model Combined (`batch=1`)** | Backbone + Head + Softmax (`50` warm + `200` timed, no I/O) | **`235.09 ± 5.93 ms`** | Median **`234.54 ms`**, Min `206.55 ms`, P95 `246.09 ms`, P99 `249.38 ms`, Max `249.66 ms` (**`4.25 FPS`**) |
| **End-to-End From Disk** | Decode + Preprocess + Model + `argmax` (`200` JPEGs) | **`254.78 ± 7.12 ms`** | Median `254.95 ms`, Min `240.71 ms`, P95 `264.90 ms`, P99 `275.81 ms`, Max `292.37 ms` (**`3.92 FPS`**) |
| **Full-Val Batched Pass** | `batch_size=64` over `2,489` images (`evaluation` run) | **`587.52 s`** (**`236.0 ms/img`**) | Essentially identical per-image cost to `batch=1` (`235.09 ms`) — confirms **compute/GEMM saturation on CPU** even at batch 1 |

### 4.3 Cross-Model Baseline Context vs MobileNetV2 (Same Host, Audited Controlled Runs)

| Model Variant | Format / Runtime | Accuracy (`2,489` Val) | Mel. Recall | Size | Pure Inference (`batch=1`) | Relative vs DINOv2 |
|---|---|---:|---:|---:|---:|---|
| **MobileNetV2 Full INT8** | TFLite (`XNNPACK`, controlled `ONEDNN=1`)* | `90.72%` | `86.73%` | `2.94 MB` | **`4.51 ± 0.20 ms`** | **52.1× faster**, **112.3× smaller** |
| **MobileNetV2 Dynamic TFLite** | TFLite (`XNNPACK`, controlled) | `90.88%` | `85.62%` | `2.74 MB` | **`27.76 ± 1.77 ms`** | **8.5× faster**, **120.4× smaller** |
| **MobileNetV2 FP32** | Keras (`TF 2.21.0`, controlled `ONEDNN=1`) | `90.96%` | `87.06%` | `30.21 MB` | **`104.89 ± 9.95 ms`** | **2.2× faster**, **10.9× smaller** |
| **DINOv2 ViT-B/14 (Baseline)** | PyTorch `2.10.0+cpu` (oneAPI MKL / MKL-DNN) | **`86.66%`** | **`79.09%`** | **`330.34 MB`** | **`235.09 ± 5.93 ms`** | `1.0×` baseline |

*\*Historical note: `mobilenetv2_int8_verification_report.md` initially recorded `16.35 ms` INT8 / `269.56 ms` FP32 under higher host contention and `TF_ENABLE_ONEDNN_OPTS=0`; controlled back-to-back runs in `mobilenetv2_pruning_independent_audit.md` measured `4.51 ms` INT8 / `104.89 ms` FP32 (`ONEDNN=1`). Even against the worst-case loaded INT8 timing (`16.35 ms`), DINOv2 FP32 is **14.4× slower**.*

---

## 5. Technical Optimization Opportunity Analysis (Analysis Only — Not Executed)

Because **`99.96%`** of inference latency and **`99.997%`** of model parameters reside inside **`dinov2_vitb14`** — specifically its `48` transformer `nn.Linear` projections (`85.02M` params = **98.20%** of total weights) — optimizing [`dinov2_100_epoch_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) alone is meaningless (`2,307` params, `59 µs`, `11.4 KB`). Every realistic deployment optimization must target the **`dinov2_vitb14` ViT-B/14 backbone** jointly with the head.

### 5.1 Dynamic INT8 Quantization (PyTorch Eager vs ONNX Runtime)

- **Head-Only Dynamic INT8:** Would reduce `2,307` params (~`9 KB` -> ~`2.3 KB`) while leaving `99.997%` of weights (`330.33 MB`) and `99.96%` of runtime (`235.00 ms`) untouched. **No value.**
- **Full ViT-B/14 `nn.Linear` Dynamic INT8 (`85.02M` params, `98.20%` of model):**
  - *Expected Footprint Reduction:* Compresses `qkv`, `proj`, `fc1`, `fc2`, and `classifier` weights from `float32` to `int8`, cutting total serialized checkpoint size from **`~330.3 MB` -> `~85–90 MB`** (`~73%` size reduction, ~`3.7–3.9×` compression) while keeping `patch_embed`, `LayerNorm`, `LayerScale`, `pos_embed`, and `SDPA` attention matmuls in FP32.
  - *Critical PyTorch `2.10+cpu` Windows Constraint Verified in Audit:* Here `torch.backends.quantized.supported_engines == ['onednn']` (**neither `fbgemm` nor `qnnpack` is present**). In PyTorch CPU eager mode, `torch.ao.quantization.quantize_dynamic` (`dtype=torch.qint8` on `nn.Linear`) must execute runtime per-activation dynamic min-max calibration and INT8 GEMM dispatch via oneDNN. For **`batch=1` (`257` tokens)** on Intel Alder Lake (`i5-1235U`, AVX2/AVX-VNNI), eager PyTorch dynamic quant often yields **modest or variable speedups (`~1.1×–1.5×`)** because per-token quantization overhead, non-Linear ops (`LayerNorm`, `GELU`, `SDPA` QK^T/AV matmuls, bicubic `pos_embed`), and Python module dispatch remain FP32.
  - *ONNX Runtime Dynamic / Weight-Only INT8 Advantage:* Exporting to ONNX and applying `onnxruntime.quantization.quantize_dynamic` (targeting `MatMul` / `Gemm` + ORT transformer graph fusion for Multi-Head Attention, `SkipLayerNormalization`, and `FastGelu` / `BiasGelu`) typically outperforms eager PyTorch significantly (**`~1.5×–2.5×` CPU speedup** at `batch=1`, `~85–95 MB` model) with typically very small accuracy drift (`< 0.5 pp`). *Requires installing `onnx` and `onnxruntime` (+ `onnxscript` for PyTorch 2.x dynamo exporter or legacy TorchScript ONNX path).*

### 5.2 Static INT8 Post-Training Quantization (PTQ)

- Quantizing both weights and activations (`qkv`, `proj`, `fc1`, `fc2`, `patch_embed`, and optionally residual/attention matmuls) maximizes integer arithmetic (`AVX-VNNI` on i5-1235U) and removes dynamic min-max overhead.
- **ViT Calibration Risks for Clinical Dermoscopy (`HIGH SENSITIVITY CAUTION`):**
  1. **Activation Outliers:** Pre-LayerNorm ViTs with `LayerScale` (`init_values=1.0`), `GELU` asymmetry, and `[CLS]` token channel spikes exhibit skewed/heavy-tailed activation distributions; naive tensor-wise static INT8 min-max calibration can seriously degrade ViT accuracy (`2–10+ pp` drops without percentile/entropy calibration or SmoothQuant).
  2. **Attention Softmax & Residual Quantization:** Quantizing `QK^T` before softmax or `LayerScale` multipliers introduces large quantization noise, directly threatening **`Melanoma` recall (`79.09%` baseline — already `7.97 pp` below MobileNetV2's `87.06%`)**.
  3. **Calibration Hygiene:** If static INT8 calibration is attempted (e.g., via ORT static QDQ/QOperator or OpenVINO NNCF PTQ), the calibration subset (~`250–500` images) must be sampled **exclusively from the `9,956` training partition (`random_state=42`)** — zero validation or test images.

### 5.3 FP16 / BF16 Conversion on Host CPU

- **Counter-Indicated for Compute Acceleration (`NOT RECOMMENDED`):**
  - The target host (`i5-1235U`, Windows 11, `torch 2.10.0+cpu`, **no CUDA GPU**, **no AVX-512 FP16**) lacks native hardware FP16 vector FMA units in PyTorch's CPU kernels. Running FP16 inference on CPU frequently triggers slow scalar fallbacks, errors on unsupported ops (`LayerNorm` / bicubic `grid_sampler`), or FP16 -> FP32 upcasting per op, making inference **slower** than FP32 while only halving disk/weight memory (`330 MB -> ~165 MB`). (BF16 is likewise emulated without AVX-512 BF16 / AMX on Alder Lake-U.)

### 5.4 Static-Shape Graph Cleaning & ONNX Runtime Optimization (`HIGH-VALUE STRUCTURAL PRE-STEP`)

Inspecting the actual DINOv2 forward source revealed two immediate, **zero-accuracy-loss** architectural cleanups for fixed `224 × 224` deployment before/alongside ONNX export:

1. **Freeze/Fold `interpolate_pos_encoding()` (`vision_transformer.py:182-214`):** Currently every `224 × 224` forward pass executes dynamic shape math and `F.interpolate(..., mode='bicubic', antialias=False)` (`scale_factor` with `offset=0.1`) on `pos_embed [1, 1370, 768] -> [1, 257, 768]`. Pre-computing `[1, 257, 768]` numerically identically at `224 × 224` eliminates bicubic interpolation from the runtime graph AND cuts `pos_embed` by **`854,784` parameters (`-3.26 MB` FP32)**. (Also drop unused `mask_token` [`768` params] and fold single-module backbone + `DINOv2LinearClassifier` into one exportable `nn.Module`.)
2. **`MemEffAttention` Clean Export (`attention.py:82-88`):** Since `xformers` is absent, inference already uses standard PyTorch `F.scaled_dot_product_attention`, which exports cleanly to ONNX **opset ≥ 17** (recommended for native `LayerNormalization` and clean SDPA decomposition/fusion) without custom CUDA/xFormers ops.
3. **`torch.compile` Caveat:** On Windows CPU (`MSVC`, no Triton), eager `oneDNN`/MKL or ORT/OpenVINO deployment is generally far cleaner and faster cold-start than Inductor compilation.

### 5.5 OpenVINO IR / NNCF Weight-Only Compression (`STRONGEST X86 CPU CANDIDATE IF INSTALLED`)

- Because this host uses a **12th Gen Intel Core i5-1235U** (Alder Lake, **AVX2 + AVX-VNNI**), **OpenVINO Runtime + NNCF INT8 weight compression / PTQ** is specifically engineered for Intel CPUs and ViT architectures (`MatMul` / MHSA / GELU fusion + weight-only INT8 or INT4 compression with minimal accuracy loss and native static-shape optimization).
- *Requirement:* Neither `openvino` nor `onnx`/`onnxruntime` is currently installed; installing either requires environment modification/approval.

### 5.6 Summary Comparison Matrix of Viable Phase 2 Paths

| Candidate Strategy | Requires New Packages? | Expected Model Size | Expected CPU Speedup vs `235 ms` | Accuracy / Melanoma Risk | Priority Assessment |
|---|---|---:|---:|---|---|
| **PyTorch Eager Dynamic INT8 (`onednn`)** | **No** (works out of the box) | **~85–90 MB** (`-73%`) | Low–Modest (`~1.05×–1.4×`, must benchmark `onednn`!) | Very Low (`<0.5 pp`) | **Best zero-dependency baseline experiment** (`100%` self-contained in existing `.venv`) |
| **Static `224×224` ONNX FP32 + ORT Graph Fusion** | **Yes** (`onnx`, `onnxruntime`) | `~327 MB` (folded `pos_embed`) | Modest (`~1.2×–1.6×`) | **Zero** (`~0.00 pp`, bit-close to PyTorch) | **Strong clean base** for quantized ORT & unified single-file `.onnx` |
| **ONNX Runtime Dynamic INT8 (`MatMul`/`Gemm`)** | **Yes** (`onnx`, `onnxruntime`) | **~83–88 MB** (`-74%`) | Modest–High (`~1.5×–2.5×`) | Low (`<0.5–1.0 pp`) | **High-value cross-platform CPU target** |
| **OpenVINO FP32 + INT8 Weight Compression / PTQ** | **Yes** (`openvino`, `nncf`) | **~83–90 MB** (WC INT8) | High on Intel i5-1235U (`~2×–3.5×`) | Low (WC) / Moderate (Static PTQ) | **Best raw Intel x86 CPU throughput option** |
| **Edge Re-Architecture (ViT-S/14 Distillation)** | Training required | `~85 MB` FP32 / `~22 MB` INT8 | `~3.5×–4×` (before INT8) | Requires retraining | Out of scope for PTQ/deployment optimization unless `<30 MB` Raspberry Pi target is mandated |

> **Deployment Architecture Strategic Note:** Because [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py) currently executes **MobileNetV2 TFLite only** (`4.51 ms`, `2.94 MB`) while PyTorch DINOv2 (`235.09 ms`, `330.34 MB`) is offline/unwired, compressing/accelerating DINOv2 primarily unblocks deploying the **weighted MobileNetV2 + DINOv2 ensemble** (`0.5/0.5 = 91.36%` accuracy on `2,489` val vs `90.96%` MobileNetV2 alone) on desktop/server CPU hosts under a practical sub-150 ms budget. Even after `75%` compression (`~85 MB`), DINOv2 ViT-B/14 remains **~29× larger** and significantly slower than MobileNetV2 Full INT8 (`2.94 MB`, `4.51 ms`).

---

## 6. Strict Preservation & Immutability Compliance Verification (`VERIFIED`)

All **18** monitored baseline checkpoints, scripts, evaluation reports, cached hub files, and serialized `.npy`/`.csv`/`.json` artifacts were cryptographically hashed (`SHA-256`) before evaluation began and re-verified after all benchmarks and JSON generators completed (**0 mutations, 100% hash + byte-size + timestamp match**):

| Preserved File Path | Size (Bytes) | Modified (`ISO`) | Verified SHA-256 Digest (Pre-Audit = Post-Audit) | Match |
|---|---:|---|---|:---:|
| [`models/dinov2_100_epoch_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) | `11,669` | `2026-08-26T22:57:47` | `783e31fb03a60a178b81724c70db5878b107389be228f4c5ced2f5e61c88d725` | **100%** |
| [`models/dinov2_classifier_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth) | `11,677` | `2026-08-26T22:50:22` | `e40f3d5d6f01451af7a87807cae2046cfaba1cc18ceef847dc64044b29d82796` | **100%** |
| [`dinov2_vitb14_pretrain.pth`](file:///C:/Users/HP/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth) | `346,378,731` | `2026-08-26T23:28:05` | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` | **100%** |
| [`models/best_mobilenetv2.keras`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) | `31,673,490` | `2026-08-17T13:57:56` | `ebbd575e3d7b340d8d806515abc918cc5260b8cd4b8ef4c41c93f9582ebf33d4` | **100%** |
| [`models/best_mobilenetv2.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.tflite) | `2,877,224` | `2026-08-29T14:47:36` | `acf368659274be0fca0bfd6d4715f9a515e7d701d44592aee33fae4e73b2b0eb` | **100%** |
| [`models/best_mobilenetv2_full_int8.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) | `3,083,384` | `2026-11-10T08:11:58` | `00924d9d127fd73438ba526ac6bbe28ac3fdc12ee3fff4272212aa82ae075da7` | **100%** |
| [`models/best_mobilenetv2_pruned.keras`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_pruned.keras) | `31,673,383` | `2026-11-10T09:22:07` | `0364f6c4d152c7ccf6d555aadd49eb82bf293f52983bb29bbc73eee4acc3814f` | **100%** |
| [`evaluation.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluation.py) | `20,755` | `2026-08-28T17:41:47` | `55b7c934ebad767c0208e77c5300a314d78ab0e68883846acfeae0e813a8a78b` | **100%** |
| [`evaluate_final_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_final_ensemble.py) | `11,293` | `2026-08-28T17:41:47` | `3bf83cd176265da896c2ad3b1c901271a051d205839eac8a457e5aef20e0b854` | **100%** |
| [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py) | `91,483` | `2026-08-31T11:03:05` | `925f7505bd7b726d94b0763589f9e22aeaca039f0ead7dae2806507e16bbe08d` | **100%** |
| [`verify_dinov2.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/verify_dinov2.py) | `3,076` | `2026-08-27T00:08:49` | `900f2134495729409bd00335f942a97aee1b6f25e0773c3c5b00a449d406da8a` | **100%** |
| [`verify_ensemble.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/verify_ensemble.py) | `3,809` | `2026-08-27T00:41:44` | `6672d6689cb63dfe826a77fcb530f2f6a772e5b12371b10e739b0ba7618390f0` | **100%** |
| [`sweep_weights.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/sweep_weights.py) | `3,273` | `2026-08-27T01:02:24` | `507399cfcc4d9b97354b8945aa145c5fb4af40e4002e3be5c953dd5d7d688f0d` | **100%** |
| [`metrics/final_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/final_metrics.json) | `1,428` | `2026-08-27T09:28:11` | `732806ebc9ed6087ef2ec22705c0e7467d3c4b72b92d31632d2aaca1d58b0fa0` | **100%** |
| [`metrics/dinov2_classification_report.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/dinov2_classification_report.csv) | `484` | `2026-08-27T09:28:08` | `115016868a7418446bf7dd3aeac0f78fb29a524f54c8377d47bc37bebdbafa0e` | **100%** |
| [`metrics/dinov2_val_probabilities.npy`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/dinov2_val_probabilities.npy) | `29,996` | `2026-11-09T12:53:21` | `da0a29d3dbecf3d049a31517376beb91b2486ce4766cc56fbf193ec9b35e932d` | **100%** |
| [`metrics/val_ground_truth.npy`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/metrics/val_ground_truth.npy) | `10,084` | `2026-11-09T12:53:21` | `dae9a29b4f694ac1ea0e04c8d0dd7adba6b563d7cc34f87392807cee66e6bb71` | **100%** |
| [`results/dinov2_epoch_history.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2_epoch_history.csv) | `1,546` | `2026-08-28T17:36:56` | `200952011861f66ca8fbe392ec0bbd6b91c0dadb16242ac63529e3eabdb0c004` | **100%** |

Full verification records are also stored in [`results/dinov2/dinov2_baseline_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_baseline_metrics.json).

---

## Phase 1 Gate — Awaiting Decision Before Any Phase 2 Optimization

Phase 1 verification is complete and **no optimization has been applied**. Recommended next step options for Phase 2 (all strictly non-destructive, saving artifacts separately under `models/` and `results/dinov2/`):

1. **Zero-Dependency Baseline (`PyTorch 2.10` native `onednn`):** Merge `dinov2_vitb14` + `dinov2_100_epoch_best.pth` (`pos_embed` folded to `[1, 257, 768]`) and benchmark PyTorch Dynamic INT8 (`qint8` on `48+1` Linear layers) vs FP32 on the `2,489` validation images.
2. **ONNX Runtime Deployment Path *(recommended if `onnx` + `onnxruntime` installation is approved)*:** Export static `224 × 224` folded FP32 ONNX (`opset >= 17`), apply ORT transformer graph optimization + Dynamic INT8 (`MatMul`/`Gemm`), and optionally compare calibration-safe INT8 PTQ (`9,956` train-only calibration).
3. **Intel CPU Peak Speedup Path *(if `openvino` installation is approved)*:** Convert folded DINOv2 to OpenVINO IR + INT8 weight compression (`nncf`) tuned for `i5-1235U`.

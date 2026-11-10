# DermaVision — Final GitHub Repository Preparation & File Inventory

**Audit Date:** `2026-11-10`  
**Repository Branch / Remote:** `main` (`origin -> https://github.com/d-chetan-kumar/dermaVision-Hardware-.git`)  
**Workspace:** [`c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019 dataset)`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset))  
**Submission Mode:** Clean working-tree preparation only (**no `git commit` or `git push` executed**).

---

## 1. Configuration Verification (`.gitignore`, `requirements.txt`, `README.md`)

### 1.1 [`.gitignore`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/.gitignore) Verification
Replaced legacy blanket `*.csv` rule with raw-dataset-specific patterns so reproducible experiment CSVs, histories, split manifests, Grad-CAM tables, and final comparison matrices can be tracked safely while heavy/sensitive files remain ignored:
- **Dataset Exclusions:** `ISIC_2019_Training_Input/`, `dermavision-isic-2019/`, `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`, `ISIC_2019_*.csv`.
- **Large / Redundant Model Exclusions:** `models/*.keras`, `models/**/*.keras`, `models/*.h5`, `models/dinov2/*.pth`, `results/pruning/checkpoints/`, `*.h5`, `*.pb`, `dinov2_vitb14_pretrain.pth`.
  - *Note:* Existing tracked [`models/best_mobilenetv2.keras`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) (`30.21 MiB`), [`models/best_mobilenetv2.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.tflite) (`2.74 MiB`), [`models/dinov2_100_epoch_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) (`11.67 KB`), and [`models/dinov2_classifier_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth) (`11.68 KB`) remain tracked by Git, while redundant `.keras`/`.h5` checkpoints and `91.63 MB` DINOv2 INT8 weights are ignored. New [`models/best_mobilenetv2_full_int8.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) (`2.94 MiB`) is unignored and ready to stage.
- **Packaging, Temp, Cache, Venv & Secrets Exclusions:** `dermavision_kaggle_pack/`, `dermavision_kaggle_pack.zip`, `scratch/`, `scratch_*.png`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.keras_cache/`, `.env`, `.env.*`, `.venv`, `.venv/`, `venv/`, `.streamlit/secrets.toml`, `.streamlit/credentials.toml`, `secrets.json`, `credentials.json`, `*.pem`, `*.key`, `*.log`.

### 1.2 [`requirements.txt`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/requirements.txt) Verification
AST import verification across all workspace `.py` scripts confirms complete dependency coverage:
- **Streamlit & Visualization Core:** `streamlit>=1.36.0`, `numpy>=1.26.0`, `pillow>=10.0.0`, `fpdf2>=2.7.0`, `scikit-learn>=1.4.0`, `pandas>=2.1.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`, `tqdm>=4.66.0`, `psutil>=5.9.0`.
- **Edge Runtime (Linux / ARM64 Marker):** `ai-edge-litert; platform_system == "Linux"` (prevents pip wheel failures on Windows development hosts while enabling lightweight Raspberry Pi / Linux Streamlit Cloud execution).
- **Research, Quantization, Pruning, DINOv2 & Grad-CAM:** `tensorflow>=2.16.0`, `tf-keras>=2.16.0`, `tensorflow-model-optimization>=0.8.0`, `torch>=2.2.0`, `torchvision>=0.17.0`, `nbformat>=5.9.0`.

### 1.3 [`README.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/README.md) Update Summary
Updated with current architecture diagram preview (`architecture_diagram_final.{png,svg}`), `ISIC 2019` (`12,445` images) split and `54.04%` lesion-overlap disclosure, two-branch topology details, consolidated 7-model metrics table (`i5-1235U` paired Run B & Phase 2 conditions), MobileNetV2 Full INT8 vs DINOv2 Dynamic INT8 collapse explanation, 50% pruning failure summary, Grad-CAM nested pre-softmax summary across all 4 clean non-overlapping samples (explicitly stating heatmaps do not prove clinical explainability), deployment recommendation ([`best_mobilenetv2_full_int8.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) + [`prediction_int8.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py)), and Git LFS policy.

---

## 2. Large Files (`> 100 MB` & `50–100 MB`) and Git LFS Requirements

Local `git-lfs` CLI is installed (`git-lfs/3.7.1`), though `.gitattributes` is not yet configured because **zero files inside the staged/trackable repository payload exceed `50 MB` or `100 MB`**:

| Asset / Path | Exact Size (`Bytes` / `MB` / `MiB`) | Tier | Current Git Status | Git LFS Requirement |
|---|---:|---|---|---|
| `ISIC_2019_Training_Input/` (`25,333` files) | `9,801,452,265 B` (`9.801 GB` / `9.128 GiB`) | **`> 100 MB` (dir)** | **Ignored** (`.gitignore`) | Never commit raw dataset to Git/GitHub. |
| `~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` *(external cache)* | `346,378,731 B` (`346.38 MB` / `330.33 MiB`) | **`> 100 MB`** | **External** (+ ignored if copied) | **Requires Git LFS** if ever added to repo (`git lfs track "models/dinov2/dinov2_vitb14_pretrain.pth"`). Otherwise downloaded via Torch Hub cache. |
| `results/pruning/checkpoints/mobilenetv2_pruned_epoch_01..05.h5` (`5` files) | `41,810,616 B` each (`209.05 MB` / `199.37 MiB` total) | **`> 100 MB` (dir)** | **Ignored** (`.gitignore`) | Intermediate pruning epochs; excluded cleanly. |
| `models/dinov2/dinov2_vitb14_dynamic_int8.pth` | `91,627,892 B` (`91.63 MB` / `87.38 MiB`) | `50–100 MB` warn | **Ignored** (`.gitignore`) | Rejected `-25.07 pp` INT8 collapse weight. If ever committed, track exact path in LFS: `git lfs track "models/dinov2/dinov2_vitb14_dynamic_int8.pth"`. |
| `dinov2_vitb14_folded_experimental_int8.pth` *(scratch only)* | `88,211,872 B` (`88.21 MB` / `84.13 MiB`) | `50–100 MB` warn | **Not in repo** (scratch) | Experimental ablation only; not promoted to `models/`. |
| [`models/revision_v1/mobilenetv2/best_mobilenetv2.keras`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/revision_v1/mobilenetv2/best_mobilenetv2.keras) | `31,679,109 B` (`31.68 MB` / `30.21 MiB`) | `10–50 MB` | **Ignored** (`.gitignore`) | Placeholder/duplicate Revision V1 copy; excluded. |
| [`models/best_mobilenetv2.keras`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.keras) | `31,673,490 B` (`31.67 MB` / `30.21 MiB`) | `10–50 MB` | **Already Tracked** | Below GitHub `50 MB` warn / `100 MB` block. *(Optional future `git lfs migrate import --include="models/*.keras"`.)* |
| `models/best_mobilenetv2_pruned.keras` | `31,673,383 B` (`31.67 MB` / `30.21 MiB`) | `10–50 MB` | **Ignored** (`.gitignore`) | Rejected `-13.61 pp` Mel recall checkpoint; excluded. |
| [`results/gradcam/gradcam_comparison.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_comparison.png) | `9,552,739 B` (`9.55 MB` / `9.11 MiB`) | `< 10 MB` | **Ready to Add** (`??`) | Standard Git (`300 DPI` paper figure, under `10 MB`). |
| [`models/best_mobilenetv2_full_int8.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) | `3,083,384 B` (`3.08 MB` / `2.94 MiB`) | `< 5 MB` | **Ready to Add** (`??`) | Standard Git (recommended compact edge winner). |
| [`models/best_mobilenetv2.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2.tflite) | `2,877,224 B` (`2.88 MB` / `2.74 MiB`) | `< 5 MB` | **Already Tracked** | Standard Git. |

> **Important Git LFS Rule for `.pth`:** Never add a wildcard `*.pth filter=lfs` rule unless migrating existing tracked files, because [`models/dinov2_100_epoch_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_100_epoch_best.pth) (`11,669 B`) and [`models/dinov2_classifier_best.pth`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2_classifier_best.pth) (`11,677 B`) are already regular-Git tracked. Use explicit file paths in `.gitattributes` if LFS is ever enabled.

---

## 3. Privacy, Hygiene & Cleanliness Verification

Automated inspection confirmed:
- **Dataset Exclusion:** `0` ISIC 2019 `.jpg` files (`ISIC_2019_Training_Input/`) or raw dataset CSVs (`ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`) appear in tracked (`git ls-files`) or untracked (`??`) state.
- **Virtual Environments & IDE Files:** `.venv`, `venv/`, `.vscode/`, `.idea/` exclusions intact; `0` venv artifacts staged or untracked.
- **Credentials & Secrets Scan:** Regex audit across all `138+` text/source/JSON/CSV/Markdown files returned **`0` hits** for API keys, tokens, private keys, or credentials.
- **Unnecessary Caches & Archives Excluded:** `.keras_cache/` (`9.52 MB`), `__pycache__/` (`0.21 MB`), `models/dinov2/__pycache__/`, `dermavision-isic-2019/` (`0.03 MB` stray bare git dir), `dermavision_kaggle_pack/` (`1.64 MB`), `dermavision_kaggle_pack.zip` (`0.25 MB`), and `scratch/` are all cleanly ignored.

---

## 4. Complete GitHub File Inventory

### 4.1 Modified Tracked Files to Update (`4` files)

| File | Status | Purpose of Update |
|---|---|---|
| [`.gitignore`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/.gitignore) | `M` | Replace blanket `*.csv` ignore with raw-ISIC-only CSV ignores; ignore heavy pruning `.h5`, redundant `.keras`, `91.63 MB` DINOv2 `.pth`, `.keras_cache/`, and Kaggle packs. |
| [`requirements.txt`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/requirements.txt) | `M` | Align all AST-verified dependencies with `ai-edge-litert; platform_system == "Linux"` plus TensorFlow/TFLite, TFMOT pruning, PyTorch/Torchvision, tqdm, psutil, nbformat. |
| [`README.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/README.md) | `M` | Updated with final diagrams, 2-branch topology, leakage audit, 7-model benchmark table, quantization & pruning findings, Grad-CAM results, deployment recommendation, and Git LFS guide. |
| [`streamlit_app.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/streamlit_app.py) | `M` | Pre-existing local UI diff referencing `Picture1-C.png` and `Picture2-C.png` for DINOv2 and MobileNetV2 training history plots. |

### 4.2 New / Untracked Files to Add (`94` files, `34.50 MiB` / `36.18 MB` total)

#### A. Grad-CAM Scripts, Metrics, Reports & Visualizations (`23` files)
- [`gradcam.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py) (`7,108 B`) — Nested two-stage replay & pre-softmax Grad-CAM utility.
- [`gradcam.py.bak`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/gradcam.py.bak) (`2,006 B`) — Byte-exact backup of original 62-line prototype.
- [`results/gradcam/generate_gradcam.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/generate_gradcam.py) (`17,564 B`) — Reproducible clean-slice Grad-CAM generator script.
- [`results/gradcam/gradcam_results.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_results.csv) (`3,645 B`) — Full quantitative metadata, probabilities, logits & stats for Cases 1–4.
- [`results/gradcam/gradcam_audit_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_audit_report.md) (`16,630 B`) — Read-only pre-generation Grad-CAM audit.
- [`results/gradcam/gradcam_generation_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_generation_report.md) (`8,764 B`) — Final Grad-CAM generation & visual inspection report.
- [`results/gradcam/gradcam_comparison.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/gradcam/gradcam_comparison.png) (`9,552,739 B` / `9.11 MiB`) — Master `300 DPI` 4×3 comparison grid.
- `results/gradcam/gradcam_samples/` (`16` files: `_original.png`, `_heatmap.png`, `_overlay.png`, `_panel.png` for `01_bcc_tp_ISIC_0073082`, `02_melanoma_tp_ISIC_0033522`, `03_nevus_tp_ISIC_0026951`, `04_melanoma_fn_as_nevus_ISIC_0025313`).

#### B. Final Consolidated Comparison & Repository Inventory (`4` files)
- [`results/final/final_model_comparison.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/final_model_comparison.csv) (`9,576 B`)
- [`results/final/final_model_comparison.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/final_model_comparison.json) (`42,473 B`)
- [`results/final/final_model_comparison_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/final_model_comparison_report.md) (`38,589 B`)
- [`results/final/github_file_inventory.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/final/github_file_inventory.md) (this inventory)

#### C. MobileNetV2 Edge Deployment, Quantization & Pruning (`12` files)
- [`models/best_mobilenetv2_full_int8.tflite`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/best_mobilenetv2_full_int8.tflite) (`3,083,384 B` / `2.94 MiB`) — **Recommended Edge Deployment Model**.
- [`prediction_int8.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prediction_int8.py) (`5,346 B`) — Standalone Full INT8 TFLite inference wrapper.
- [`quantize_and_evaluate_int8.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/quantize_and_evaluate_int8.py) (`30,055 B`)
- [`quantize_mobilenetv2_int8.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/quantize_mobilenetv2_int8.py) (`7,047 B`)
- [`prune_and_train_mobilenetv2.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/prune_and_train_mobilenetv2.py) (`19,708 B`)
- [`evaluate_pruned_mobilenetv2.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/evaluate_pruned_mobilenetv2.py) (`35,747 B`)
- [`results/int8/mobilenetv2_int8_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/int8/mobilenetv2_int8_metrics.json) (`4,234 B`)
- [`results/int8/mobilenetv2_int8_verification_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/int8/mobilenetv2_int8_verification_report.md) (`7,710 B`)
- [`results/pruning/mobilenetv2_pruning_history.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_history.csv) (`1,220 B`)
- [`results/pruning/mobilenetv2_pruning_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_metrics.json) (`11,236 B`)
- [`results/pruning/mobilenetv2_pruning_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_report.md) (`12,051 B`)
- [`results/pruning/mobilenetv2_pruning_independent_audit.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/pruning/mobilenetv2_pruning_independent_audit.md) (`49,185 B`)

#### D. DINOv2 Phase 1 & Phase 2 Experiment Artifacts (`7` files)
- [`models/dinov2/dinov2_dynamic_int8_inference.py`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/models/dinov2/dinov2_dynamic_int8_inference.py) (`10,591 B`)
- [`results/dinov2/dinov2_baseline_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_baseline_metrics.json) (`33,289 B`)
- [`results/dinov2/dinov2_deployment_baseline_audit.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_deployment_baseline_audit.md) (`52,648 B`)
- [`results/dinov2/dinov2_dynamic_int8_metrics.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_metrics.json) (`78,275 B`)
- [`results/dinov2/dinov2_dynamic_int8_report.md`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_dynamic_int8_report.md) (`19,071 B`)
- [`results/dinov2/dinov2_layer_parameter_inventory.json`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2/dinov2_layer_parameter_inventory.json) (`95,720 B`)
- [`results/dinov2_epoch_history.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/dinov2_epoch_history.csv) (`1,546 B`)

#### E. Architecture Diagrams, App Charts & Manuscript Tables (`9` files)
- [`architecture_diagram_final.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram_final.png) (`1,459,093 B` / `1.39 MiB`)
- [`architecture_diagram_final.svg`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram_final.svg) (`72,482 B`)
- [`architecture_diagram_revised.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram_revised.png) (`1,418,284 B` / `1.35 MiB`)
- [`architecture_diagram_revised.svg`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/architecture_diagram_revised.svg) (`72,159 B`)
- [`Picture1-C.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/Picture1-C.png) (`311,179 B` — used by `streamlit_app.py`)
- [`Picture2-C.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/Picture2-C.png) (`417,678 B` — used by `streamlit_app.py`)
- [`melanoma_candidates.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/melanoma_candidates.png) (`726,192 B`) & [`predictions_grid.png`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/predictions_grid.png) (`770,445 B`) — pre-existing root diagnostic grids referenced in Grad-CAM audit
- [`DermaVision_IEEE_Table_III_Hyperparameters.docx`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/DermaVision_IEEE_Table_III_Hyperparameters.docx) (`38,131 B`)

#### F. Metrics Arrays, CSVs & Ensemble Diagnostics (`15` files under `metrics/` & `results/`)
- `metrics/`: `classification_report.csv`, `dinov2_classification_report.csv`, `dinov2_val_probabilities.npy`, `ensemble_04_06_classification_report.csv`, `ensemble_04_06_confusion_matrix.csv`, `ensemble_04_06_confusion_matrix.png`, `ensemble_04_06_evaluation.txt`, `ensemble_04_06_metrics.json`, `ensemble_04_06_predictions.npy`, `ensemble_04_06_probabilities.npy`, `ensemble_classification_report.csv`, `ensemble_weight_comparison.csv`, `mobilenet_val_probabilities.npy`, `val_ground_truth.npy` (`14` compact files, `108 B – 134 KB` each)
- [`results/best_ensemble_classification_report.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/best_ensemble_classification_report.csv) (`486 B`)

#### G. Audits, Epoch Logs, Revision V1 Splits & Prepared Training Scripts (`24` files)
- [`reviewer_comment_response_matrix.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/reviewer_comment_response_matrix.csv) (`26,748 B`)
- [`results/lesion_overlap_audit.txt`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/lesion_overlap_audit.txt) (`8,380 B`)
- [`results/mobilenetv2_epoch_history.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/mobilenetv2_epoch_history.csv) (`1,549 B`)
- [`results/validation_image_order.csv`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/results/validation_image_order.csv) (`96,589 B`)
- [`data/splits/revision_v1/`](file:///c:/Users/HP/OneDrive/Desktop/Dermavision(ISIC2019%20dataset)/data/splits/revision_v1): `split_summary.json`, `train.csv` (`1.02 MB`), `val.csv` (`220 KB`), `test.csv` (`217 KB`)
- `results/revision_v1/`: `mobilenetv2/training_config.json`, `mobilenetv2/training_history.csv`, `mobilenetv2_corrected/batch_distribution_audit.json`, `mobilenetv2_corrected/pretraining_verification_report.md`, `mobilenetv2_corrected/training_config.json`, `mobilenetv2_training_diagnostic_report.md`, `performance_investigation_report.md`, plus `results/revision_v1_split_report.txt`, `results/revision_v1_training_technical_review.txt`
- Prepared Revision V1 training/Kaggle helpers: `DermaVision_Revision_V1_Training_Kaggle.ipynb`, `build_kaggle_notebook.py`, `create_kaggle_pack.py`, `import_kaggle_results.py`, `train_dinov2_revision_v1.py`, `train_mobilenetv2_revision_v1.py`, `train_mobilenetv2_revision_v1_1.py`

### 4.3 Already Tracked Repository Files (`45` files total, `35.60 MiB`)
*(Includes all `4` modified files listed in Section 4.1 plus `41` unmodified tracked files: `.python-version`, `architecture_diagram.html`, `benchmark.py`, `convert_to_tflite.py`, `evaluate_final_ensemble.py`, `evaluation.py`, `generate_all_graphs.py`, `generate_dinov2_graphs.py`, `graphs/*.png` (`7` files), `metrics/` (`12` previously tracked PNGs/JSONs), `models/best_mobilenetv2.keras`, `models/best_mobilenetv2.tflite`, `models/class_names.json`, `models/dinov2_100_epoch_best.pth`, `models/dinov2_classifier_best.pth`, `prediction.py`, `preprocessing.py`, `scratch_verify_ckpt.py`, `sweep_weights.py`, `test_pipeline.py`, `three_components.html`, `utils.py`, `validator.py`, `verify_dinov2.py`, `verify_ensemble.py`.)*

---

## 5. Manual Git Commands for the User (Not Run Automatically)

Review changes first and stage/commit manually when ready:

```bash
# 1. Inspect working-tree delta
git status -su

# 2. Stage all prepared updates and new files
git add -A

# 3. Verify staged files contain zero ISIC images, venvs, caches, or >100 MB weights
git status -s

# 4. Commit and push manually when ready
git commit -m "Prepare DermaVision repo: 7-model comparison, Full INT8, pruning, DINOv2 audits, Grad-CAM & README"
git push origin main
```

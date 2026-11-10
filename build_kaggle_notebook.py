import os
import json
import zlib
import base64
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# 1. Read and compress the Revision V1 splits for embedded fallback
with open('data/splits/revision_v1/train.csv', 'rb') as f:
    tr_b64 = base64.b64encode(zlib.compress(f.read(), 9)).decode('ascii')
with open('data/splits/revision_v1/val.csv', 'rb') as f:
    va_b64 = base64.b64encode(zlib.compress(f.read(), 9)).decode('ascii')

nb = new_notebook()

# Cell 1: Markdown Title & Instructions
c1_md = """# DermaVision Revision V1.1 – GPU Accelerated Training Pipeline (Kaggle)
### Lesion-Grouped Cross-Validation: MobileNetV2 + DINOv2 ViT-B/14 Soft Ensemble

> **Time Estimate:** ~12–15 minutes total on Kaggle GPU (Tesla T4 or P100)  
> **Classes (3-Class ISIC 2019):** `0: BCC`, `1: Melanoma`, `2: Nevus`  
> **Partitions:** Train (8,712 images), Validation (1,875 images) — **Strict zero lesion overlap**

---
### Quick Setup Checklist:
1. **Accelerator:** In the right sidebar, set **Accelerator** $\\rightarrow$ **GPU T4 x2** or **GPU P100**.
2. **Internet:** Set **Internet** $\\rightarrow$ **On** (needed for downloading ImageNet and DINOv2 ViT-B/14 weights).
3. **Data:** Add any Kaggle ISIC 2019 dataset (e.g., `isic-2019`, `andrewmvd/isic-2019`, or search `ISIC 2019`).
4. Click **Run All** (or `Shift + Enter` through the cells).
"""
nb.cells.append(new_markdown_cell(c1_md))

# Cell 2: Code - GPU Check and Environment Setup
c2_code = """# Cell 1: Environment & GPU Verification
import os
import sys
import time
import json
import zlib
import base64
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Set random seeds for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("="*60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"PyTorch Version:    {torch.__version__}")
gpus_tf = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPUs:    {gpus_tf}")
print(f"CUDA Available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected! Please set Accelerator to GPU in the right sidebar for fast ~12-minute execution.")
print("="*60)
"""
nb.cells.append(new_code_cell(c2_code))

# Cell 3: Code - Auto-detect ISIC dataset and manifests
c3_code = f"""# Cell 2: Dataset Discovery & Manifest Preparation
# Embedded Revision V1 split data (fallback if manifests not uploaded)
EMBEDDED_TRAIN_B64 = "{tr_b64}"
EMBEDDED_VAL_B64 = "{va_b64}"

OUTPUT_DIR = Path("/kaggle/working/dermavision_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Unpack manifests or locate existing CSVs
train_csv_path = None
val_csv_path = None

# Search in /kaggle/input for manifests
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower() == "train.csv" and "revision_v1" in root.lower():
            train_csv_path = os.path.join(root, f)
        elif f.lower() == "val.csv" and "revision_v1" in root.lower():
            val_csv_path = os.path.join(root, f)

if not train_csv_path or not val_csv_path:
    print("Extracting verified Revision V1 manifests from embedded payload...")
    train_csv_path = str(OUTPUT_DIR / "train.csv")
    val_csv_path = str(OUTPUT_DIR / "val.csv")
    with open(train_csv_path, "wb") as f:
        f.write(zlib.decompress(base64.b64decode(EMBEDDED_TRAIN_B64)))
    with open(val_csv_path, "wb") as f:
        f.write(zlib.decompress(base64.b64decode(EMBEDDED_VAL_B64)))

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

print(f"Loaded Train Manifest: {{len(train_df):,}} images")
print(f"Loaded Val Manifest:   {{len(val_df):,}} images")

# 2. Index all image files in /kaggle/input
print("\\nIndexing ISIC image paths in /kaggle/input (this takes ~3-5 seconds)...")
image_path_map = {{}}
input_root = "/kaggle/input"
for root, _, files in os.walk(input_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_id = os.path.splitext(f)[0]
            full_path = os.path.join(root, f)
            image_path_map[img_id] = full_path
            image_path_map[f] = full_path

print(f"Total image files indexed: {{len(image_path_map):,}}")

# Map paths in train_df and val_df
train_df["image_path"] = train_df["image_id"].map(image_path_map)
val_df["image_path"] = val_df["image_id"].map(image_path_map)

# Check missing images
missing_train = train_df["image_path"].isna().sum()
missing_val = val_df["image_path"].isna().sum()

if missing_train > 0 or missing_val > 0:
    print(f"WARNING: Missing images found: Train missing={{missing_train}}, Val missing={{missing_val}}")
    print("Please ensure an ISIC 2019 dataset (with images like ISIC_0000002.jpg) is added to your Kaggle notebook.")
else:
    print("SUCCESS: 100% of Train and Validation image paths resolved successfully!")
    print(f"Sample resolved path: {{train_df['image_path'].iloc[0]}}")
"""
nb.cells.append(new_code_cell(c3_code))

# Cell 4: Code - MobileNetV2 Data Pipeline with Revision V1.1 Shuffling Fix & Augmentation
c4_code = """# Cell 3: MobileNetV2 Revision V1.1 tf.data Pipeline
# Fixes data streaming defect: Full in-memory shuffle + 8,712 buffer size + clinical augmentations

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        tf.keras.layers.RandomRotation(factor=0.125, fill_mode="reflect", name="aug_rotation"),  # +/- 45 deg
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", name="aug_zoom")
    ], name="clinical_augmentation")

def build_tf_dataset(df, batch_size=64, is_training=True, seed=42):
    if is_training:
        # Step 1: Pre-shuffle DataFrame in-memory to dismantle contiguous class blocks
        shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        shuffled_df = df.reset_index(drop=True)

    image_paths = shuffled_df["image_path"].values
    labels = shuffled_df["class_label"].values.astype(np.int32)

    def _parse_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0  # Historical [0, 1] scaling
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        # Step 2: Full dataset shuffle buffer for true IID stochastic batches
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_layer = get_augmentation_layer()
        def _augment(img, label):
            return aug_layer(img, training=True), label
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = build_tf_dataset(train_df, batch_size=BATCH_SIZE, is_training=True)
val_ds = build_tf_dataset(val_df, batch_size=BATCH_SIZE, is_training=False)

# Audit first 5 batches to verify mixed class distributions
print("Auditing first 5 batches for class mixing:")
for idx, (_, labels) in enumerate(train_ds.take(5)):
    counts = pd.Series(labels.numpy()).value_counts().to_dict()
    print(f"  Batch {idx+1}: BCC(0)={counts.get(0, 0):2d}, Melanoma(1)={counts.get(1, 0):2d}, Nevus(2)={counts.get(2, 0):2d}")
"""
nb.cells.append(new_code_cell(c4_code))

# Cell 5: Code - MobileNetV2 Model Architecture and Two-Stage Staged Training
c5_code = """# Cell 4: MobileNetV2 Two-Stage Staged Training Pipeline
# Stage 1: Warmup (10 epochs, frozen backbone, lr=5e-4)
# Stage 2: Fine-Tuning (40 epochs, backbone unfrozen, lr=1e-5, BatchNorm frozen)

def build_mobilenetv2_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = base_model.output  # 1280-d
    x = tf.keras.layers.Dense(512, activation='relu', name='dense_512')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name='DermaVision_MobileNetV2_V1_1')
    return model, base_model

mb_model, mb_base = build_mobilenetv2_model()

# Custom callback for comprehensive epoch-wise metrics tracking
class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, val_labels, ckpt_path):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_labels = val_labels
        self.ckpt_path = ckpt_path
        self.best_val_acc = 0.0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Predict on validation
        preds_list = []
        for images, _ in self.val_dataset:
            preds_list.append(self.model(images, training=False).numpy())
        y_prob = np.concatenate(preds_list, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        val_acc = accuracy_score(self.val_labels, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(self.val_labels, y_pred, average='macro', zero_division=0)
        _, per_rec, _, _ = precision_recall_fscore_support(self.val_labels, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        record = {
            "epoch": epoch + 1,
            "train_loss": logs.get("loss", 0.0),
            "train_acc": logs.get("accuracy", 0.0),
            "val_loss": logs.get("val_loss", 0.0),
            "val_acc": val_acc,
            "macro_f1": f1,
            "bcc_rec": per_rec[0],
            "mel_rec": per_rec[1],
            "nev_rec": per_rec[2],
            "lr": float(self.model.optimizer.learning_rate.numpy())
        }
        self.history.append(record)

        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.model.save(self.ckpt_path)

        star = " ★ BEST" if is_best else ""
        print(f"Ep {epoch+1:02d} | TrLoss: {record['train_loss']:.4f} TrAcc: {record['train_acc']*100:.2f}% | "
              f"ValLoss: {record['val_loss']:.4f} ValAcc: {val_acc*100:.2f}% | "
              f"MacF1: {f1:.4f} (BCC:{per_rec[0]*100:.1f}%, Mel:{per_rec[1]*100:.1f}%, Nev:{per_rec[2]*100:.1f}%){star}")

val_labels = val_df["class_label"].values.astype(np.int32)
mb_ckpt_path = str(OUTPUT_DIR / "best_mobilenetv2_rev_v1_1.keras")
metrics_cb = ComprehensiveMetricsCallback(val_ds, val_labels, mb_ckpt_path)

# ==============================================================
# STAGE 1: Classification Head Warmup (10 Epochs)
# ==============================================================
print("\\n" + "="*70)
print("STAGE 1: Classification Head Warmup (Epochs 1-10, lr=5e-4, frozen backbone)")
print("="*70)
mb_base.trainable = False
mb_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

t0 = time.time()
mb_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[metrics_cb],
    verbose=0
)
print(f"Stage 1 completed in {(time.time() - t0)/60:.2f} minutes.")

# ==============================================================
# STAGE 2: Backbone Fine-Tuning (40 Epochs, lr=1e-5, frozen BatchNorm)
# ==============================================================
print("\\n" + "="*70)
print("STAGE 2: Backbone Fine-Tuning (Epochs 11-50, lr=1e-5, frozen BatchNorm)")
print("="*70)
mb_base.trainable = True
# Freeze all BatchNorm layers for stable moving stats
for layer in mb_base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

mb_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

t1 = time.time()
mb_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    initial_epoch=10,
    callbacks=[metrics_cb],
    verbose=0
)
print(f"Stage 2 completed in {(time.time() - t1)/60:.2f} minutes.")
print(f"MobileNetV2 Best Validation Accuracy: {metrics_cb.best_val_acc*100:.2f}%")

# Save history
mb_history_df = pd.DataFrame(metrics_cb.history)
mb_history_df.to_csv(OUTPUT_DIR / "mobilenetv2_history.csv", index=False)
"""
nb.cells.append(new_code_cell(c5_code))

# Cell 6: Code - MobileNetV2 Validation Probabilities
c6_code = """# Cell 5: Generate & Save MobileNetV2 Validation Probabilities
print("Loading best MobileNetV2 checkpoint for evaluation...")
best_mb_model = tf.keras.models.load_model(mb_ckpt_path)

mb_probs_list = []
for images, _ in val_ds:
    mb_probs_list.append(best_mb_model(images, training=False).numpy())
mb_val_probs = np.concatenate(mb_probs_list, axis=0)
np.save(OUTPUT_DIR / "mobilenet_val_probabilities.npy", mb_val_probs)

mb_preds = np.argmax(mb_val_probs, axis=1)
mb_acc = accuracy_score(val_labels, mb_preds)
print(f"MobileNetV2 Validation Top-1 Accuracy: {mb_acc*100:.2f}%")
print("\\nMobileNetV2 Classification Report:")
print(classification_report(val_labels, mb_preds, target_names=CLASS_NAMES, digits=4))
"""
nb.cells.append(new_code_cell(c6_code))

# Cell 7: Code - DINOv2 Feature Extraction and Linear Classifier Training
c7_code = """# Cell 6: DINOv2 ViT-B/14 Pipeline (Feature Extraction + Linear Classifier)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

print("Loading DINOv2 ViT-B/14 backbone via torch.hub...")
dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dinov2_backbone.eval()
dinov2_backbone.to(device)

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DINOImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["image_path"].values
        self.labels = df["class_label"].values.astype(np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def extract_dinov2_features(backbone, df, desc="Extracting Features"):
    ds = DINOImageDataset(df, transform=dino_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    all_feats = []
    all_lbls = []
    t_start = time.time()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone(imgs)  # 768-d
            all_feats.append(feats.cpu())
            all_lbls.append(lbls)
    print(f"  {desc} completed in {(time.time() - t_start):.1f}s")
    return torch.cat(all_feats, dim=0), torch.cat(all_lbls, dim=0)

print("\\nExtracting 768-d features for Train and Validation partitions...")
train_feats, train_labels_pt = extract_dinov2_features(dinov2_backbone, train_df, "Train Features")
val_feats, val_labels_pt = extract_dinov2_features(dinov2_backbone, val_df, "Validation Features")

# Train Linear Classifier on cached 768-d features (100 Epochs)
class DINOv2LinearClassifier(nn.Module):
    def __init__(self, in_features=768, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)

train_tensor_ds = TensorDataset(train_feats, train_labels_pt)
val_tensor_ds = TensorDataset(val_feats, val_labels_pt)
train_loader = DataLoader(train_tensor_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_tensor_ds, batch_size=64, shuffle=False)

classifier = DINOv2LinearClassifier(768, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

best_dino_acc = 0.0
dino_ckpt_path = str(OUTPUT_DIR / "best_dinov2_rev_v1.pth")
dino_history = []

print("\\n" + "="*70)
print("Training DINOv2 Linear Classifier (100 Epochs on Cached GPU Tensors)...")
print("="*70)
t_dino = time.time()

for epoch in range(1, 101):
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = classifier(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct += (out.argmax(dim=1) == y_b).sum().item()
        total += len(y_b)

    tr_loss = total_loss / total
    tr_acc = correct / total

    # Validation
    classifier.eval()
    val_preds, val_targets, val_probs_list = [], [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            out = classifier(x_b)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            val_probs_list.append(probs)
            val_preds.extend(out.argmax(dim=1).cpu().numpy())
            val_targets.extend(y_b.numpy())

    v_acc = accuracy_score(val_targets, val_preds)
    _, _, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
    _, per_rec, _, _ = precision_recall_fscore_support(val_targets, val_preds, average=None, labels=[0, 1, 2], zero_division=0)

    dino_history.append({
        "epoch": epoch,
        "train_loss": tr_loss,
        "train_acc": tr_acc,
        "val_acc": v_acc,
        "macro_f1": f1,
        "bcc_rec": per_rec[0],
        "mel_rec": per_rec[1],
        "nev_rec": per_rec[2]
    })

    if v_acc > best_dino_acc:
        best_dino_acc = v_acc
        torch.save(classifier.state_dict(), dino_ckpt_path)

    if epoch % 10 == 0 or epoch == 1:
        print(f"DINOv2 Ep {epoch:03d} | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:.2f}% | ValAcc: {v_acc*100:.2f}% | MacF1: {f1:.4f}")

print(f"DINOv2 Linear Head trained in {(time.time() - t_dino):.1f}s. Best Val Acc: {best_dino_acc*100:.2f}%")
pd.DataFrame(dino_history).to_csv(OUTPUT_DIR / "dinov2_history.csv", index=False)

# Compute & Save DINOv2 validation probabilities using best checkpoint
classifier.load_state_dict(torch.load(dino_ckpt_path))
classifier.eval()
with torch.no_grad():
    dino_val_probs = torch.softmax(classifier(val_feats.to(device)), dim=1).cpu().numpy()
np.save(OUTPUT_DIR / "dinov2_val_probabilities.npy", dino_val_probs)
"""
nb.cells.append(new_code_cell(c7_code))

# Cell 8: Code - Soft Probability Weighted Ensemble
c8_code = """# Cell 7: Soft Voting Ensemble (0.4 MobileNetV2 + 0.6 DINOv2)
print("="*70)
print("EVALUATING DERMAVISION REVISION V1 ENSEMBLE (0.4 MobileNetV2 + 0.6 DINOv2)")
print("="*70)

W_MB = 0.40
W_DINO = 0.60

ens_val_probs = (W_MB * mb_val_probs) + (W_DINO * dino_val_probs)
ens_val_preds = np.argmax(ens_val_probs, axis=1)

ens_acc = accuracy_score(val_labels, ens_val_preds)
prec, rec, f1, _ = precision_recall_fscore_support(val_labels, ens_val_preds, average='macro', zero_division=0)
_, per_rec, _, _ = precision_recall_fscore_support(val_labels, ens_val_preds, average=None, labels=[0, 1, 2], zero_division=0)

print(f"Ensemble Validation Accuracy:   {ens_acc*100:.2f}%")
print(f"Ensemble Macro-Precision:       {prec*100:.2f}%")
print(f"Ensemble Macro-Recall:          {rec*100:.2f}%")
print(f"Ensemble Macro-F1 Score:        {f1*100:.2f}%")
print(f"  - BCC Recall (0):             {per_rec[0]*100:.2f}%")
print(f"  - Melanoma Recall (1):        {per_rec[1]*100:.2f}%")
print(f"  - Nevus Recall (2):           {per_rec[2]*100:.2f}%")

print("\\nEnsemble Classification Report:")
print(classification_report(val_labels, ens_val_preds, target_names=CLASS_NAMES, digits=4))

# Comparison Summary
summary_data = {
    "Model": ["MobileNetV2 (V1.1)", "DINOv2 ViT-B/14", "Weighted Ensemble (0.4/0.6)"],
    "Accuracy (%)": [round(mb_acc*100, 2), round(best_dino_acc*100, 2), round(ens_acc*100, 2)],
    "Macro F1": [
        round(precision_recall_fscore_support(val_labels, mb_preds, average='macro')[2], 4),
        round(precision_recall_fscore_support(val_labels, np.argmax(dino_val_probs, axis=1), average='macro')[2], 4),
        round(f1, 4)
    ],
    "BCC Recall (%)": [
        round(precision_recall_fscore_support(val_labels, mb_preds, average=None)[1][0]*100, 2),
        round(precision_recall_fscore_support(val_labels, np.argmax(dino_val_probs, axis=1), average=None)[1][0]*100, 2),
        round(per_rec[0]*100, 2)
    ],
    "Melanoma Recall (%)": [
        round(precision_recall_fscore_support(val_labels, mb_preds, average=None)[1][1]*100, 2),
        round(precision_recall_fscore_support(val_labels, np.argmax(dino_val_probs, axis=1), average=None)[1][1]*100, 2),
        round(per_rec[1]*100, 2)
    ],
    "Nevus Recall (%)": [
        round(precision_recall_fscore_support(val_labels, mb_preds, average=None)[1][2]*100, 2),
        round(precision_recall_fscore_support(val_labels, np.argmax(dino_val_probs, axis=1), average=None)[1][2]*100, 2),
        round(per_rec[2]*100, 2)
    ]
}
summary_df = pd.DataFrame(summary_data)
print("\\n" + "="*70)
print(summary_df.to_string(index=False))
print("="*70)
summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)
"""
nb.cells.append(new_code_cell(c8_code))

# Cell 9: Code - Visualizations and Publication Figures
c9_code = """# Cell 8: Generate Publication Figures & Curves
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Training Curves (MobileNetV2 and DINOv2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MobileNetV2 Loss & Acc
axes[0, 0].plot(mb_history_df["epoch"], mb_history_df["train_loss"], label="Train Loss", color="royalblue", lw=2)
axes[0, 0].plot(mb_history_df["epoch"], mb_history_df["val_loss"], label="Val Loss", color="crimson", lw=2)
axes[0, 0].axvline(10, color="gray", linestyle="--", label="Stage 2 Unfreeze")
axes[0, 0].set_title("MobileNetV2: Training & Validation Loss", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()

axes[0, 1].plot(mb_history_df["epoch"], mb_history_df["train_acc"]*100, label="Train Acc", color="royalblue", lw=2)
axes[0, 1].plot(mb_history_df["epoch"], mb_history_df["val_acc"]*100, label="Val Acc", color="forestgreen", lw=2)
axes[0, 1].axvline(10, color="gray", linestyle="--", label="Stage 2 Unfreeze")
axes[0, 1].set_title("MobileNetV2: Training & Validation Accuracy", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend()

# DINOv2 Loss & Acc
dino_h_df = pd.DataFrame(dino_history)
axes[1, 0].plot(dino_h_df["epoch"], dino_h_df["train_loss"], label="Train Loss", color="darkorange", lw=2)
axes[1, 0].set_title("DINOv2 Linear Probe: Cross-Entropy Loss", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()

axes[1, 1].plot(dino_h_df["epoch"], dino_h_df["train_acc"]*100, label="Train Acc", color="darkorange", lw=2)
axes[1, 1].plot(dino_h_df["epoch"], dino_h_df["val_acc"]*100, label="Val Acc", color="darkmagenta", lw=2)
axes[1, 1].set_title("DINOv2 Linear Probe: Accuracy", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy (%)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300)
plt.show()

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(val_labels, ens_val_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(7, 6))
labels_annot = np.array([
    [f"{count}\\n({pct*100:.1f}%)" for count, pct in zip(row_c, row_p)]
    for row_c, row_p in zip(cm, cm_norm)
])
sns.heatmap(cm_norm, annot=labels_annot, fmt='', cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=True, ax=ax)
ax.set_title("DermaVision Revision V1 Ensemble Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
ax.set_ylabel("True Ground Truth Class", fontsize=11, labelpad=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ensemble_confusion_matrix.png", dpi=300)
plt.show()

# 3. Per-Class Recall Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
classes = ["BCC", "Melanoma", "Nevus"]
recalls = [per_rec[0]*100, per_rec[1]*100, per_rec[2]*100]
colors = ["#2b5c8f", "#d95f02", "#7570b3"]
bars = ax.bar(classes, recalls, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("Ensemble Per-Class Sensitivity (Recall)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sensitivity / Recall (%)", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ensemble_per_class_recalls.png", dpi=300)
plt.show()
print("All publication figures successfully plotted and saved!")
"""
nb.cells.append(new_code_cell(c9_code))

# Cell 10: Code - Auto-package results for 1-Click Download
c10_code = """# Cell 9: Create 1-Click Downloadable Results Archive
zip_archive_path = "/kaggle/working/dermavision_revision_v1_results"
shutil.make_archive(zip_archive_path, 'zip', OUTPUT_DIR)
print("="*70)
print("DERMAVISION REVISION V1 TRAINING COMPLETE!")
print(f"Results archived to: {zip_archive_path}.zip")
print("="*70)
print("You can download 'dermavision_revision_v1_results.zip' directly from the Kaggle 'Data' / 'Output' sidebar on the right.")
"""
nb.cells.append(new_code_cell(c10_code))

# Save the notebook
output_nb_path = "DermaVision_Revision_V1_Training_Kaggle.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully created notebook: {output_nb_path}")


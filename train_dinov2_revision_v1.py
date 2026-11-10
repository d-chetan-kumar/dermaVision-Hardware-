"""
train_dinov2_revision_v1.py
================================================================================
DermaVision Revision V1: DINOv2 ViT-B/14 Baseline Training Pipeline
--------------------------------------------------------------------------------
Trains a 3-class linear diagnostic classifier on top of the self-supervised
DINOv2 ViT-B/14 foundation model using the Revision V1 lesion-level manifests.

Features:
- Pre-trained DINOv2 ViT-B/14 foundation backbone (768-d embedding)
- Frozen backbone baseline strategy matching original experiment
- DINOv2LinearClassifier head (nn.Linear(768, 3))
- Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Deterministic feature extraction and caching (--cache_features / --use_cached_features)
    * Precomputes 768-d embeddings for train and val into isolated .pt files
    * Reduces 100-epoch training time from ~25 hours to ~15 seconds on CPU
    * Strictly isolates train and validation; prohibited from touching test data
- Optional balanced class weighting (--use_class_weights)
- Comprehensive epoch-wise metric tracking:
    * Loss & Overall Accuracy
    * Macro Precision, Recall, and F1-Score
    * Per-class Recalls: BCC Recall, Melanoma Recall, Nevus Recall
- Strict test set isolation (test.csv is prohibited from being loaded)
- Checkpoint saving (best_dinov2.pth with full architecture & split metadata)
- SHA-256 manifest integrity verification
- Reproducible random seeds
- CLI smoke-test mode for forward-pass and pipeline verification
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}

def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file for cryptographic provenance."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_manifest(manifest_path, partition_name):
    """Verifies that the manifest exists, is structurally valid, and is NOT test.csv."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"{partition_name} manifest not found at: {path}")
    if "test" in path.name.lower():
        raise PermissionError(
            f"SECURITY VIOLATION: Test manifest '{path.name}' detected in training pipeline! "
            "Test data must remain strictly isolated until final frozen model evaluation."
        )
    df = pd.read_csv(path)
    required_cols = ["image_id", "image_path", "class_name", "class_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Manifest '{path.name}' missing required column: {col}")
    return df

class ISICRevisionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_path"].values
        self.labels = self.df["class_label"].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class DINOv2LinearClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

def get_transforms():
    """Deterministic transforms for train and validation using standard ImageNet parameters."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return eval_transform, eval_transform

def extract_and_cache_features(backbone, df, output_dir, partition_name, device, batch_size=64):
    """
    Extracts 768-d features from the frozen DINOv2 backbone and caches them to disk.
    Strictly isolated: train and validation are cached separately.
    Test set is prohibited from ever being passed to this function.
    """
    if "test" in partition_name.lower():
        raise PermissionError(f"SECURITY VIOLATION: Caching features for test data is strictly prohibited!")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / f"{partition_name}_features.pt"
    labels_path = output_dir / f"{partition_name}_labels.pt"

    print(f"\n[Feature Extraction] Caching {partition_name} features ({len(df):,} samples)...")
    transform, _ = get_transforms()
    dataset = ISICRevisionDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    backbone.eval()
    all_features = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            feats = backbone(images).cpu()
            all_features.append(feats)
            all_labels.append(labels)
            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - start_time
                done_samples = min((i + 1) * batch_size, len(df))
                print(f"  Processed {done_samples:,}/{len(df):,} [{done_samples/len(df)*100:.1f}%] in {elapsed:.1f}s")

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    torch.save(features_tensor, features_path)
    torch.save(labels_tensor, labels_path)
    print(f"Saved {partition_name} features: {features_path} (Shape: {features_tensor.shape})")
    print(f"Saved {partition_name} labels:   {labels_path} (Shape: {labels_tensor.shape})")
    return features_tensor, labels_tensor

def compute_evaluation_metrics(y_true, y_pred):
    """Computes overall accuracy, macro precision/recall/F1, and per-class recalls."""
    acc = float(np.mean(y_pred == y_true))
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "acc": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "bcc_recall": float(recalls[0]),
        "melanoma_recall": float(recalls[1]),
        "nevus_recall": float(recalls[2])
    }

def run_smoke_test(backbone, classifier, train_df, val_df, device, batch_size=4):
    print("\n" + "="*70)
    print("RUNNING DINOV2 PIPELINE SMOKE TEST (FORWARD-PASS & INTEGRITY CHECK)")
    print("="*70)
    print(f"1. Device:                 {device}")
    print(f"2. Train manifest samples: {len(train_df)}")
    print(f"3. Val manifest samples:   {len(val_df)}")

    for p in list(train_df["image_path"][:3]) + list(val_df["image_path"][:2]):
        assert os.path.exists(p), f"Image file not found: {p}"
    print("4. Sample image files verified on disk.")

    train_tf, val_tf = get_transforms()
    train_ds = ISICRevisionDataset(train_df.head(batch_size * 2), transform=train_tf)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    images, labels = next(iter(loader))
    print(f"5. Batch image tensor shape: {images.shape} (Expected: ({batch_size}, 3, 224, 224))")
    print(f"6. Batch label tensor shape: {labels.shape} (Expected: ({batch_size},))")
    assert images.shape == (batch_size, 3, 224, 224)
    assert labels.shape == (batch_size,)

    images = images.to(device)
    labels = labels.to(device)

    # Forward pass backbone
    with torch.no_grad():
        features = backbone(images)
    print(f"7. Backbone features shape:  {features.shape} (Expected: ({batch_size}, 768))")
    assert features.shape == (batch_size, 768)

    # Forward pass classifier
    logits = classifier(features)
    print(f"8. Classifier logits shape:  {logits.shape} (Expected: ({batch_size}, 3))")
    assert logits.shape == (batch_size, 3)

    probs = torch.softmax(logits, dim=1)
    sums = torch.sum(probs, dim=1).cpu().detach().numpy()
    assert np.allclose(sums, 1.0, atol=1e-4), "Softmax output does not sum to 1.0!"
    print("9. Softmax probabilities sum to 1.0 verified.")

    # Single optimization step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer.zero_grad()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"10. Forward-backward gradient update verified. Single-batch loss: {loss.item():.4f}")

    # Verify metrics computation
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert "macro_f1" in metrics and "bcc_recall" in metrics and "melanoma_recall" in metrics
    print(f"11. Comprehensive metrics computation verified. Mini Acc: {metrics['acc']*100:.2f}% | Macro-F1: {metrics['macro_f1']*100:.2f}%")

    print("\n[SMOKE TEST PASSED] DINOv2 ViT-B/14 Revision V1 pipeline is fully operational!")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="DermaVision Revision V1: DINOv2 ViT-B/14 Baseline Training Pipeline")
    parser.add_argument("--train_manifest", type=str, default="data/splits/revision_v1/train.csv", help="Path to train CSV")
    parser.add_argument("--val_manifest", type=str, default="data/splits/revision_v1/val.csv", help="Path to val CSV")
    parser.add_argument("--output_model_dir", type=str, default="models/revision_v1/dinov2_vitb14", help="Model checkpoint directory")
    parser.add_argument("--output_result_dir", type=str, default="results/revision_v1/dinov2_vitb14", help="History & log directory")
    parser.add_argument("--cache_dir", type=str, default="results/revision_v1/dinov2_vitb14/cached_features", help="Directory for cached embeddings")
    parser.add_argument("--cache_features", action="store_true", help="Extract and cache features to disk before training")
    parser.add_argument("--use_cached_features", action="store_true", help="Train directly from cached features on disk")
    parser.add_argument("--use_class_weights", action="store_true", help="Apply balanced inverse class weights to loss function")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    parser.add_argument("--smoke_test", action="store_true", help="Run forward-pass smoke test and exit without full training")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("DERMAVISION: DINOV2 VIT-B/14 TRAINING PIPELINE (REVISION V1)")
    print("="*70)
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"Execution Device:    {device}")
    print(f"Random Seed:         {args.seed}")
    print(f"Learning Rate:       {args.lr}")
    print(f"Weight Decay:        {args.weight_decay}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Max Epochs:          {args.epochs}")
    print(f"Use Cached Features: {args.use_cached_features}")
    print(f"Use Class Weights:   {args.use_class_weights}")

    # Load and verify manifests
    train_df = verify_manifest(args.train_manifest, "Training")
    val_df = verify_manifest(args.val_manifest, "Validation")

    train_hash = compute_sha256(args.train_manifest)
    val_hash = compute_sha256(args.val_manifest)

    print(f"Train Manifest:      {args.train_manifest} ({len(train_df):,} samples) | SHA-256: {train_hash[:16]}...")
    print(f"Val Manifest:        {args.val_manifest} ({len(val_df):,} samples) | SHA-256: {val_hash[:16]}...")

    # Class distribution & weighting
    train_counts = train_df["class_label"].value_counts().sort_index()
    print("Training class distribution:")
    for idx, cname in enumerate(CLASS_NAMES):
        count = train_counts.get(idx, 0)
        pct = (count / len(train_df)) * 100
        print(f"  Class {idx} ({cname:8s}): {count:,} samples ({pct:.2f}%)")

    class_weights_tensor = None
    if args.use_class_weights:
        weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df["class_label"].values)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print("Computed balanced class weights:")
        for idx, cname in enumerate(CLASS_NAMES):
            print(f"  Class {idx} ({cname}): {float(class_weights_tensor[idx]):.4f}")

    # Load DINOv2 ViT-B/14 backbone
    print("\nLoading pre-trained DINOv2 ViT-B/14 backbone from PyTorch Hub...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    # Classifier head
    classifier = DINOv2LinearClassifier(feature_dim=768, num_classes=3).to(device)

    if args.smoke_test:
        run_smoke_test(backbone, classifier, train_df, val_df, device, batch_size=4)
        print("Smoke test complete. Exiting without full training as requested.")
        return

    # Check directories
    os.makedirs(args.output_model_dir, exist_ok=True)
    os.makedirs(args.output_result_dir, exist_ok=True)

    # Handle feature caching request
    if args.cache_features:
        extract_and_cache_features(backbone, train_df, args.cache_dir, "train", device, batch_size=args.batch_size)
        extract_and_cache_features(backbone, val_df, args.cache_dir, "val", device, batch_size=args.batch_size)
        meta = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "feature_dim": 768,
            "train_manifest_sha256": train_hash,
            "val_manifest_sha256": val_hash,
            "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(Path(args.cache_dir) / "cache_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        print(f"[Caching Complete] Features saved in: {args.cache_dir}")
        print("You can now train rapidly using: --use_cached_features")
        return

    # Save training config
    config = {
        "model": "DINOv2_ViT-B/14",
        "strategy": "Frozen_Backbone_Linear_Probing",
        "dataset_split": "revision_v1",
        "train_manifest": args.train_manifest,
        "train_manifest_sha256": train_hash,
        "val_manifest": args.val_manifest,
        "val_manifest_sha256": val_hash,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "use_class_weights": args.use_class_weights,
        "use_cached_features": args.use_cached_features,
        "epochs": args.epochs,
        "image_size": [224, 224, 3],
        "device": str(device),
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = Path(args.output_result_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to: {config_path}")

    # Build DataLoaders (cached tensors or live images)
    if args.use_cached_features:
        train_feat_path = Path(args.cache_dir) / "train_features.pt"
        train_lbl_path = Path(args.cache_dir) / "train_labels.pt"
        val_feat_path = Path(args.cache_dir) / "val_features.pt"
        val_lbl_path = Path(args.cache_dir) / "val_labels.pt"

        if not (train_feat_path.exists() and val_feat_path.exists()):
            raise FileNotFoundError(f"Cached features not found in {args.cache_dir}! Run with --cache_features first.")

        print(f"Loading cached features from: {args.cache_dir}...")
        train_feats = torch.load(train_feat_path, map_location="cpu")
        train_lbls = torch.load(train_lbl_path, map_location="cpu")
        val_feats = torch.load(val_feat_path, map_location="cpu")
        val_lbls = torch.load(val_lbl_path, map_location="cpu")

        train_ds = TensorDataset(train_feats, train_lbls)
        val_ds = TensorDataset(val_feats, val_lbls)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        is_cached_mode = True
    else:
        train_tf, val_tf = get_transforms()
        train_ds = ISICRevisionDataset(train_df, transform=train_tf)
        val_ds = ISICRevisionDataset(val_df, transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        is_cached_mode = False

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(args.output_model_dir) / "best_dinov2.pth"
    csv_log_path = Path(args.output_result_dir) / "training_history.csv"

    best_val_acc = 0.0
    history_records = []

    start_time = time.time()
    print(f"\nStarting DINOv2 linear head training for {args.epochs} epochs (Mode: {'Cached Tensors' if is_cached_mode else 'Live Images'})...")
    for epoch in range(1, args.epochs + 1):
        # Training loop
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_data, labels in train_loader:
            labels = labels.to(device)

            if is_cached_mode:
                features = batch_data.to(device)
            else:
                with torch.no_grad():
                    features = backbone(batch_data.to(device))

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation loop
        classifier.eval()
        val_loss = 0.0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_data, labels in val_loader:
                labels = labels.to(device)
                if is_cached_mode:
                    features = batch_data.to(device)
                else:
                    features = backbone(batch_data.to(device))

                logits = classifier(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                val_total += labels.size(0)
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        epoch_val_loss = val_loss / val_total
        all_val_preds = torch.cat(all_val_preds).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()

        val_metrics = compute_evaluation_metrics(all_val_labels, all_val_preds)
        epoch_val_acc = val_metrics["acc"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% | "
            f"Val F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"BCC: {val_metrics['bcc_recall']*100:.2f}% Mel: {val_metrics['melanoma_recall']*100:.2f}% Nev: {val_metrics['nevus_recall']*100:.2f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "macro_f1": val_metrics["macro_f1"],
            "bcc_recall": val_metrics["bcc_recall"],
            "melanoma_recall": val_metrics["melanoma_recall"],
            "nevus_recall": val_metrics["nevus_recall"]
        }
        history_records.append(record)
        pd.DataFrame(history_records).to_csv(csv_log_path, index=False)

        # Save checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": classifier.state_dict(),
                "feature_dim": 768,
                "num_classes": 3,
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "model_type": "DINOv2LinearClassifier",
                "backbone": "DINOv2 ViT-B/14",
                "best_val_accuracy": best_val_acc,
                "best_val_macro_f1": val_metrics["macro_f1"],
                "best_epoch": epoch,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "use_class_weights": args.use_class_weights,
                "train_manifest_sha256": train_hash,
                "val_manifest_sha256": val_hash
            }, checkpoint_path)
            print(f"  --> Checkpoint saved: {checkpoint_path} (Val Acc: {best_val_acc*100:.2f}%)")

    training_duration = time.time() - start_time

    # Save final model checkpoint
    final_checkpoint_path = Path(args.output_model_dir) / "final_dinov2.pth"
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "feature_dim": 768,
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "model_type": "DINOv2LinearClassifier",
        "backbone": "DINOv2 ViT-B/14",
        "final_val_accuracy": epoch_val_acc,
        "final_val_macro_f1": val_metrics["macro_f1"],
        "final_epoch": args.epochs,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "use_class_weights": args.use_class_weights,
        "train_manifest_sha256": train_hash,
        "val_manifest_sha256": val_hash
    }, final_checkpoint_path)

    # Read training history to find best epoch metrics
    history_df = pd.read_csv(csv_log_path)
    best_row = history_df.loc[history_df["val_acc"].idxmax()]

    summary = {
        "model": "DINOv2_ViT-B/14",
        "strategy": "Frozen_Backbone_Linear_Probing",
        "total_epochs": args.epochs,
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(best_row["val_acc"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_macro_precision": float(best_row["macro_precision"]),
        "best_val_macro_recall": float(best_row["macro_recall"]),
        "best_val_macro_f1": float(best_row["macro_f1"]),
        "best_val_bcc_recall": float(best_row["bcc_recall"]),
        "best_val_melanoma_recall": float(best_row["melanoma_recall"]),
        "best_val_nevus_recall": float(best_row["nevus_recall"]),
        "final_epoch": int(history_df.iloc[-1]["epoch"]),
        "final_val_accuracy": float(history_df.iloc[-1]["val_acc"]),
        "training_duration_seconds": float(training_duration),
        "training_duration_formatted": f"{training_duration//3600:.0f}h {(training_duration%3600)//60:.0f}m {training_duration%60:.1f}s",
        "best_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "history_path": str(csv_log_path)
    }

    summary_path = Path(args.output_result_dir) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nTraining complete!")
    print(f"Training duration: {summary['training_duration_formatted']}")
    print(f"Best checkpoint saved at:  {checkpoint_path} (Best Val Acc: {best_val_acc*100:.2f}%, Best Epoch: {summary['best_epoch']})")
    print(f"Final checkpoint saved at: {final_checkpoint_path} (Final Val Acc: {summary['final_val_accuracy']*100:.2f}%)")
    print(f"Training history saved at: {csv_log_path}")
    print(f"Training summary saved at: {summary_path}")

if __name__ == "__main__":
    main()

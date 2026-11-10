"""
train_mobilenetv2_revision_v1_1.py
================================================================================
DermaVision Revision V1.1: Corrected MobileNetV2 Staged Training Pipeline
--------------------------------------------------------------------------------
Fixes data-streaming defect by implementing:
1. In-memory DataFrame shuffling (sample frac=1.0, random_state=42)
2. Full dataset shuffle buffer (buffer_size=8,712) for true IID batch sampling
3. Two-stage fine-tuning:
   - Stage 1: Classification Head Warmup (Epochs 1-10, lr=5e-4, frozen backbone)
   - Stage 2: Backbone Fine-Tuning (Epochs 11-50, lr=1e-5, frozen BatchNorm)
4. Domain-appropriate training-only data augmentation (flips, +/-45 deg rotation, zoom)
5. Preserved historical [0, 1] preprocessing (/ 255.0)
6. Comprehensive epoch-wise metric tracking (Accuracy, Macro-F1, per-class recalls)
7. Strict test set isolation and SHA-256 provenance logging
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

# Ensure writable cache directory for Keras & Matplotlib
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.abspath(".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
IMG_SIZE = (224, 224)

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

def get_augmentation_layer():
    """Constructs training-only clinical dermoscopic augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        tf.keras.layers.RandomRotation(factor=0.125, fill_mode="reflect", name="aug_rotation"),  # +/- 45 degrees
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", name="aug_zoom")
    ], name="clinical_augmentation")

def build_dataset(df, batch_size=64, is_training=True, seed=42):
    """
    Constructs an optimized tf.data.Dataset pipeline with corrected full-dataset shuffling.
    - Training: In-memory pre-shuffle + buffer_size=len(df) + training augmentation
    - Validation: Deterministic, unshuffled, no augmentation
    """
    if is_training:
        # Step 1: Pre-shuffle DataFrame in-memory to prevent sequential class blocks
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
        # Step 2: Full dataset shuffle buffer ensuring true IID stochastic batches
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Step 3: Apply clinical data augmentation to training data
        aug_layer = get_augmentation_layer()
        def _augment(img, label):
            return aug_layer(img, training=True), label
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def audit_batch_distribution(dataset, num_batches=10):
    """Audits class representation across batches to verify mixed IID distribution."""
    audit_results = []
    all_bccs, all_mels, all_nevs = [], [], []

    for idx, (images, labels) in enumerate(dataset.take(num_batches)):
        lbls = labels.numpy()
        counts = pd.Series(lbls).value_counts().to_dict()
        bcc = int(counts.get(0, 0))
        mel = int(counts.get(1, 0))
        nev = int(counts.get(2, 0))
        total = len(lbls)

        record = {
            "batch_index": idx,
            "total_samples": total,
            "bcc_count": bcc,
            "bcc_pct": round(bcc / total * 100, 2),
            "melanoma_count": mel,
            "melanoma_pct": round(mel / total * 100, 2),
            "nevus_count": nev,
            "nevus_pct": round(nev / total * 100, 2),
            "all_three_classes_present": (bcc > 0 and mel > 0 and nev > 0)
        }
        audit_results.append(record)
        all_bccs.append(bcc)
        all_mels.append(mel)
        all_nevs.append(nev)

    summary = {
        "batches_audited": len(audit_results),
        "mean_bcc_per_batch": round(float(np.mean(all_bccs)), 2),
        "mean_melanoma_per_batch": round(float(np.mean(all_mels)), 2),
        "mean_nevus_per_batch": round(float(np.mean(all_nevs)), 2),
        "all_batches_have_all_classes": all(r["all_three_classes_present"] for r in audit_results),
        "batch_details": audit_results
    }
    return summary

def build_mobilenetv2_model(num_classes=3):
    """
    Constructs MobileNetV2 with ImageNet weights and custom classification head:
    Backbone: MobileNetV2 (ImageNet weights, 1.0 224, include_top=False)
    Head:
      - GlobalAveragePooling2D
      - Dense(256, activation='relu')
      - Dropout(0.2)
      - Dense(64, activation='relu')
      - Dropout(0.1)
      - Dense(num_classes, activation='softmax')
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer_1")
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        print("Loaded ImageNet pre-trained weights for MobileNetV2.")
    except Exception as e:
        print(f"[WARNING] ImageNet weights fetch failed ({type(e).__name__}). Initializing backbone without pre-trained weights for offline execution.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )

    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_2")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_dermavision_v1_1")
    return model, base_model

class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback calculating epoch-wise:
    - Overall Accuracy
    - Macro Precision, Macro Recall, Macro F1
    - Per-class Recalls: BCC Recall, Melanoma Recall, Nevus Recall
    Logs all values to console and writes to training_history.csv.
    """
    def __init__(self, val_ds, val_df, csv_path, stage_name, eval_model=None):
        super().__init__()
        self.val_ds = val_ds
        self.val_df = val_df
        self.csv_path = csv_path
        self.stage_name = stage_name
        self.eval_model = eval_model
        self.records = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.y_true = val_df["class_label"].values.astype(int)

    def set_stage(self, stage_name):
        self.stage_name = stage_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        active_model = self.eval_model if self.eval_model is not None else self.model
        y_prob = active_model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = float(np.mean(y_pred == self.y_true))
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            self.y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        macro_prec = float(np.mean(precisions))
        macro_rec = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        bcc_rec = float(recalls[0])
        mel_rec = float(recalls[1])
        nev_rec = float(recalls[2])

        lr = 0.0
        if hasattr(active_model, "optimizer") and hasattr(active_model.optimizer, "learning_rate"):
            try:
                lr = float(active_model.optimizer.learning_rate.numpy())
            except Exception:
                lr = float(active_model.optimizer.learning_rate)

        if acc > self.best_val_acc:
            self.best_val_acc = acc
            self.best_epoch = epoch + 1

        epoch_record = {
            "epoch": epoch + 1,
            "stage": self.stage_name,
            "train_loss": float(logs.get("loss", 0.0)),
            "train_acc": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_acc": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "bcc_recall": bcc_rec,
            "melanoma_recall": mel_rec,
            "nevus_recall": nev_rec,
            "lr": lr,
            "best_val_acc": self.best_val_acc
        }
        self.records.append(epoch_record)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)

        print(
            f" [{self.stage_name} | Epoch {epoch+1:02d}] "
            f"Val Acc: {acc*100:.2f}% (Best: {self.best_val_acc*100:.2f}%) | "
            f"Macro-F1: {macro_f1*100:.2f}% | "
            f"BCC: {bcc_rec*100:.2f}% | Mel: {mel_rec*100:.2f}% | Nev: {nev_rec*100:.2f}%"
        )

def run_smoke_test(train_df, val_df, batch_size=4):
    print("\n" + "="*70)
    print("RUNNING REVISION V1.1 PIPELINE SMOKE TEST (TWO-STAGE VERIFICATION)")
    print("="*70)

    # 1. Test Dataset Shuffling & Batch Distribution
    print("1. Verifying full dataset in-memory shuffle and batch streaming...")
    train_ds = build_dataset(train_df, batch_size=64, is_training=True, seed=42)
    audit = audit_batch_distribution(train_ds, num_batches=10)
    print(f"   Batches audited: {audit['batches_audited']}")
    print(f"   Mean BCC per batch:      {audit['mean_bcc_per_batch']} (Expected: ~17.14)")
    print(f"   Mean Melanoma per batch: {audit['mean_melanoma_per_batch']} (Expected: ~23.18)")
    print(f"   Mean Nevus per batch:    {audit['mean_nevus_per_batch']} (Expected: ~23.68)")
    assert audit["all_batches_have_all_classes"], "Defect: Not all batches have all 3 classes!"
    print("   [PASSED] Full dataset shuffling verified. All batches contain mixed classes.")

    # 2. Build Model & Test Stage 1 Trainable Parameters
    print("\n2. Verifying Stage 1 (Head Warmup) configuration...")
    model, base_model = build_mobilenetv2_model(num_classes=3)
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    stage1_trainable_weights = len(model.trainable_weights)
    print(f"   Stage 1 trainable weight tensors: {stage1_trainable_weights} (Expected: 6 head tensors)")
    assert stage1_trainable_weights == 6, f"Expected 6 trainable weights for head, got {stage1_trainable_weights}"
    print("   [PASSED] Backbone completely frozen for Stage 1.")

    # 3. Test Stage 1 Forward-Backward Step
    mini_train_ds = build_dataset(train_df.head(batch_size * 2), batch_size=batch_size, is_training=True)
    for imgs, lbls in mini_train_ds.take(1):
        with tf.GradientTape() as tape:
            preds = model(imgs, training=True)
            loss_val = model.loss(lbls, preds)
        grads = tape.gradient(loss_val, model.trainable_weights)
        assert len(grads) == 6 and grads[0] is not None
        print(f"   Stage 1 forward-backward pass verified. Loss: {float(loss_val):.4f}")

    # 4. Verify Stage 2 (Backbone Fine-Tuning with Frozen BatchNorm)
    print("\n3. Verifying Stage 2 (Backbone Fine-Tuning) configuration...")
    base_model.trainable = True
    frozen_bn_count = 0
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn_count += 1
    print(f"   Total BatchNormalization layers frozen: {frozen_bn_count} (Expected: 52)")
    assert frozen_bn_count == 52, f"Expected 52 frozen BN layers, got {frozen_bn_count}"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    stage2_trainable_weights = len(model.trainable_weights)
    print(f"   Stage 2 trainable weight tensors: {stage2_trainable_weights} (Expected: 58: 52 conv kernels + 6 head weights)")
    assert stage2_trainable_weights == 58, f"Expected 58 trainable weights, got {stage2_trainable_weights}"
    print("   [PASSED] Backbone trainable with all 52 BN layers strictly frozen.")

    # 5. Test Stage 2 Forward-Backward Step
    for imgs, lbls in mini_train_ds.take(1):
        with tf.GradientTape() as tape:
            preds = model(imgs, training=True)
            loss_val = model.loss(lbls, preds)
        grads = tape.gradient(loss_val, model.trainable_weights)
        assert len(grads) == 58 and grads[0] is not None
        print(f"   Stage 2 forward-backward pass verified. Loss: {float(loss_val):.4f}")

    # 6. Verify Comprehensive Metrics Callback Execution
    print("\n4. Verifying ComprehensiveMetricsCallback calculation...")
    mini_val_df = val_df.head(batch_size)
    mini_val_ds = build_dataset(mini_val_df, batch_size=batch_size, is_training=False)
    test_csv = Path(".keras_cache/test_smoke_metrics_v1_1.csv")
    cb = ComprehensiveMetricsCallback(mini_val_ds, mini_val_df, test_csv, "Stage 1: Warmup", eval_model=model)
    cb.on_epoch_end(0, logs={"loss": 0.5, "accuracy": 0.7, "val_loss": 0.6})
    assert test_csv.exists()
    res_df = pd.read_csv(test_csv)
    for c in ["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "macro_f1", "bcc_recall", "melanoma_recall", "nevus_recall"]:
        assert c in res_df.columns, f"Missing metric column: {c}"
    test_csv.unlink(missing_ok=True)
    print("   [PASSED] Comprehensive metrics calculation & CSV logging verified.")

    print("\n[SMOKE TEST PASSED] MobileNetV2 Revision V1.1 pipeline is fully verified!")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="DermaVision Revision V1.1: Corrected MobileNetV2 Staged Training Pipeline")
    parser.add_argument("--train_manifest", type=str, default="data/splits/revision_v1/train.csv", help="Path to train CSV")
    parser.add_argument("--val_manifest", type=str, default="data/splits/revision_v1/val.csv", help="Path to val CSV")
    parser.add_argument("--output_model_dir", type=str, default="models/revision_v1/mobilenetv2_corrected", help="Model checkpoint directory")
    parser.add_argument("--output_result_dir", type=str, default="results/revision_v1/mobilenetv2_corrected", help="History & log directory")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Stage 1: Head warmup epochs (default: 10)")
    parser.add_argument("--finetune_epochs", type=int, default=40, help="Stage 2: Backbone fine-tuning epochs (default: 40)")
    parser.add_argument("--warmup_lr", type=float, default=5e-4, help="Stage 1 learning rate (default: 5e-4)")
    parser.add_argument("--finetune_lr", type=float, default=1e-5, help="Stage 2 learning rate (default: 1e-5)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke_test", action="store_true", help="Run forward-pass smoke test and exit without full training")
    args = parser.parse_args()

    # Reproducibility
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("="*70)
    print("DERMAVISION: MOBILENETV2 REVISION V1.1 (CORRECTED STAGED PIPELINE)")
    print("="*70)
    print(f"TensorFlow Version:   {tf.__version__}")
    print(f"Random Seed:          {args.seed}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Stage 1 (Warmup):     {args.warmup_epochs} epochs @ lr={args.warmup_lr}")
    print(f"Stage 2 (Fine-Tune):  {args.finetune_epochs} epochs @ lr={args.finetune_lr}")
    print(f"Class Weighting:      DISABLED (Unweighted baseline)")
    print(f"Preprocessing:        div255 ([0.0, 1.0])")
    print(f"Data Augmentation:    Active on training (Flips, Rotations +/-45 deg, Zoom +/-10%)")
    print(f"BatchNorm Handling:   Frozen 52 BN layers during Stage 2")

    # Load and verify manifests
    train_df = verify_manifest(args.train_manifest, "Training")
    val_df = verify_manifest(args.val_manifest, "Validation")

    train_hash = compute_sha256(args.train_manifest)
    val_hash = compute_sha256(args.val_manifest)

    print(f"Train Manifest:       {args.train_manifest} ({len(train_df):,} samples) | SHA-256: {train_hash[:16]}...")
    print(f"Val Manifest:         {args.val_manifest} ({len(val_df):,} samples) | SHA-256: {val_hash[:16]}...")

    if args.smoke_test:
        run_smoke_test(train_df, val_df, batch_size=4)
        print("Smoke test complete. Exiting without full training as requested.")
        return

    # Check directories
    os.makedirs(args.output_model_dir, exist_ok=True)
    os.makedirs(args.output_result_dir, exist_ok=True)

    # Perform and save batch distribution audit
    print("\nAuditing training batch distribution with corrected full-dataset shuffling...")
    audit_train_ds = build_dataset(train_df, batch_size=args.batch_size, is_training=True, seed=args.seed)
    batch_audit = audit_batch_distribution(audit_train_ds, num_batches=10)
    audit_path = Path(args.output_result_dir) / "batch_distribution_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(batch_audit, f, indent=4)
    print(f"Saved batch distribution audit to: {audit_path}")
    print(f"  First 10 batches average: BCC={batch_audit['mean_bcc_per_batch']}, Mel={batch_audit['mean_melanoma_per_batch']}, Nev={batch_audit['mean_nevus_per_batch']}")

    # Save training configuration
    config = {
        "pipeline_version": "Revision_V1.1_Corrected",
        "model": "MobileNetV2",
        "dataset_split": "revision_v1",
        "train_manifest": args.train_manifest,
        "train_manifest_sha256": train_hash,
        "val_manifest": args.val_manifest,
        "val_manifest_sha256": val_hash,
        "batch_size": args.batch_size,
        "preprocessing": "div255 ([0.0, 1.0])",
        "data_augmentation": "RandomFlip(horizontal_and_vertical), RandomRotation(0.125), RandomZoom(0.1)",
        "class_weighting": "DISABLED",
        "seed": args.seed,
        "stage_1": {
            "name": "Classification_Head_Warmup",
            "epochs": args.warmup_epochs,
            "learning_rate": args.warmup_lr,
            "optimizer": "Adam",
            "backbone_trainable": False
        },
        "stage_2": {
            "name": "Backbone_Fine_Tuning",
            "epochs": args.finetune_epochs,
            "learning_rate": args.finetune_lr,
            "optimizer": "Adam",
            "backbone_trainable": True,
            "batch_normalization_frozen": True,
            "frozen_bn_layers_count": 52,
            "lr_scheduler": "ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=4, min_lr=1e-7)"
        },
        "output_model_dir": args.output_model_dir,
        "output_result_dir": args.output_result_dir,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = Path(args.output_result_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to: {config_path}")

    # Build model
    model, base_model = build_mobilenetv2_model(num_classes=3)

    # Prepare datasets
    train_ds = build_dataset(train_df, batch_size=args.batch_size, is_training=True, seed=args.seed)
    val_ds = build_dataset(val_df, batch_size=args.batch_size, is_training=False, seed=args.seed)

    # Checkpoint and history paths
    best_checkpoint_path = Path(args.output_model_dir) / "best_mobilenetv2.keras"
    final_checkpoint_path = Path(args.output_model_dir) / "final_mobilenetv2.keras"
    csv_log_path = Path(args.output_result_dir) / "training_history.csv"

    metrics_cb = ComprehensiveMetricsCallback(
        val_ds=val_ds,
        val_df=val_df,
        csv_path=csv_log_path,
        stage_name="Stage 1: Warmup"
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    # =========================================================================
    # STAGE 1: CLASSIFICATION HEAD WARMUP
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 1: CLASSIFICATION HEAD WARMUP (FROZEN BACKBONE)")
    print("="*70)
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.warmup_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print(f"Backbone frozen. Trainable parameters: {model.count_params() - base_model.count_params():,}")
    print(f"Training head for {args.warmup_epochs} warmup epochs...")

    start_time = time.time()
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        callbacks=[checkpoint_cb, metrics_cb]
    )

    # =========================================================================
    # STAGE 2: BACKBONE FINE-TUNING (FROZEN BATCHNORM)
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 2: BACKBONE FINE-TUNING (FROZEN BATCHNORMALIZATION)")
    print("="*70)
    base_model.trainable = True
    frozen_bn_count = 0
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn_count += 1
    print(f"Backbone un-frozen. All {frozen_bn_count} BatchNormalization layers frozen.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    metrics_cb.set_stage("Stage 2: Fine-Tuning")

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    print(f"Fine-tuning for {args.finetune_epochs} additional epochs (total {args.warmup_epochs + args.finetune_epochs})...")
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=args.warmup_epochs,
        epochs=args.warmup_epochs + args.finetune_epochs,
        callbacks=[checkpoint_cb, reduce_lr_cb, metrics_cb]
    )

    total_training_duration = time.time() - start_time

    # Save final model
    model.save(str(final_checkpoint_path))
    print(f"\nFinal model saved at: {final_checkpoint_path}")

    # Summary
    history_df = pd.read_csv(csv_log_path)
    best_row = history_df.loc[history_df["val_acc"].idxmax()]

    summary = {
        "pipeline": "Revision_V1.1_Corrected",
        "total_epochs": len(history_df),
        "best_epoch": int(best_row["epoch"]),
        "best_stage": str(best_row["stage"]),
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
        "training_duration_seconds": float(total_training_duration),
        "training_duration_formatted": f"{total_training_duration//3600:.0f}h {(total_training_duration%3600)//60:.0f}m {total_training_duration%60:.1f}s",
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "history_path": str(csv_log_path)
    }

    summary_path = Path(args.output_result_dir) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nTraining complete!")
    print(f"Training duration: {summary['training_duration_formatted']}")
    print(f"Best checkpoint:   {best_checkpoint_path} (Val Acc: {summary['best_val_accuracy']*100:.2f}%, Best Epoch: {summary['best_epoch']})")
    print(f"Final checkpoint:  {final_checkpoint_path} (Final Val Acc: {summary['final_val_accuracy']*100:.2f}%)")
    print(f"Summary saved at:  {summary_path}")

if __name__ == "__main__":
    main()

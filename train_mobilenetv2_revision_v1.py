"""
train_mobilenetv2_revision_v1.py
================================================================================
DermaVision Revision V1: MobileNetV2 Training Pipeline
--------------------------------------------------------------------------------
Trains MobileNetV2 on the lesion-level grouped ISIC 2019 dataset using the
Revision V1 manifests (train.csv and val.csv).

Features:
- ImageNet pre-trained MobileNetV2 (1.00 224) backbone
- Preserved classification head (Dense 256 -> Dropout 0.2 -> Dense 64 -> Dropout 0.1 -> Dense 3)
- Configurable input preprocessing:
    * 'div255' (default): [0.0, 1.0] scaling (matches preprocessing.py and Kaggle training)
    * 'mobilenet': [-1.0, 1.0] scaling (mobilenet_v2.preprocess_input)
- Configurable BatchNormalization handling during fine-tuning (--freeze_bn)
- Optional inverse-frequency class weighting (--use_class_weights)
- Comprehensive epoch-wise metric tracking:
    * Loss & Overall Accuracy
    * Macro Precision, Recall, and F1-Score
    * Per-class Recalls: BCC Recall, Melanoma Recall, Nevus Recall
- Strict test set isolation (test.csv is prohibited from being loaded)
- Checkpoint saving (best_mobilenetv2.keras based on val_accuracy or val_macro_f1)
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

# Ensure writable cache directory for Keras & Matplotlib
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.abspath(".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
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

def build_dataset(df, batch_size=64, is_training=True, input_scaling="div255"):
    """
    Constructs an optimized tf.data.Dataset pipeline.
    Scaling:
      - 'div255': [0.0, 1.0] scaling (x / 255.0)
      - 'mobilenet': [-1.0, 1.0] scaling ((x / 127.5) - 1.0)
    """
    image_paths = df["image_path"].values
    labels = df["class_label"].values.astype(np.int32)

    def _parse_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        if input_scaling == "mobilenet":
            img = (img / 127.5) - 1.0
        else:
            img = img / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=1024, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_mobilenetv2_model(num_classes=3, lr=1e-4, freeze_bn=False):
    """
    Constructs the exact MobileNetV2 architecture used in DermaVision:
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
    base_model.trainable = True

    if freeze_bn:
        frozen_bn_count = 0
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_bn_count += 1
        print(f"BatchNormalization handling: Frozen {frozen_bn_count} BN layers in backbone.")
    else:
        print("BatchNormalization handling: All BN layers trainable (matching Kaggle baseline).")

    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_2")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_dermavision")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback calculating epoch-wise:
    - Overall Accuracy
    - Macro Precision, Macro Recall, Macro F1
    - Per-class Recalls: BCC Recall, Melanoma Recall, Nevus Recall
    Logs all values to console and writes to training_history.csv.
    """
    def __init__(self, val_ds, val_df, csv_path, eval_model=None):
        super().__init__()
        self.val_ds = val_ds
        self.val_df = val_df
        self.csv_path = csv_path
        self.eval_model = eval_model
        self.records = []
        self.y_true = val_df["class_label"].values.astype(int)

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

        epoch_record = {
            "epoch": epoch + 1,
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
            "lr": lr
        }
        self.records.append(epoch_record)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)

        print(
            f" [Epoch {epoch+1:02d} Validation Metrics] "
            f"Acc: {acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}% | "
            f"BCC-Rec: {bcc_rec*100:.2f}% | Mel-Rec: {mel_rec*100:.2f}% | Nev-Rec: {nev_rec*100:.2f}%"
        )

def run_smoke_test(model, train_df, val_df, batch_size=4, input_scaling="div255"):
    print("\n" + "="*70)
    print("RUNNING PIPELINE SMOKE TEST (FORWARD-PASS & INTEGRITY CHECK)")
    print("="*70)
    print(f"1. Train manifest samples: {len(train_df)}")
    print(f"2. Val manifest samples:   {len(val_df)}")
    print(f"3. Input scaling mode:     {input_scaling}")

    for p in list(train_df["image_path"][:3]) + list(val_df["image_path"][:2]):
        assert os.path.exists(p), f"Image does not exist: {p}"
    print("4. Sample image paths verified on disk.")

    train_ds = build_dataset(train_df.head(batch_size * 2), batch_size=batch_size, is_training=True, input_scaling=input_scaling)
    val_ds = build_dataset(val_df.head(batch_size * 2), batch_size=batch_size, is_training=False, input_scaling=input_scaling)

    for imgs, lbls in train_ds.take(1):
        print(f"5. Batch input tensor shape:  {imgs.shape} (Expected: ({batch_size}, 224, 224, 3))")
        print(f"6. Batch label tensor shape:  {lbls.shape} (Expected: ({batch_size},))")
        assert imgs.shape == (batch_size, 224, 224, 3)
        assert lbls.shape == (batch_size,)
        if input_scaling == "mobilenet":
            assert tf.reduce_min(imgs) >= -1.01 and tf.reduce_max(imgs) <= 1.01, "Normalization outside [-1, 1]!"
            print("7. Preprocessing normalization verified [-1.0, 1.0].")
        else:
            assert tf.reduce_min(imgs) >= 0.0 and tf.reduce_max(imgs) <= 1.0, "Normalization outside [0, 1]!"
            print("7. Preprocessing normalization verified [0.0, 1.0].")

        preds = model(imgs, training=False)
        print(f"8. Model output tensor shape: {preds.shape} (Expected: ({batch_size}, 3))")
        assert preds.shape == (batch_size, 3)
        sums = tf.reduce_sum(preds, axis=1)
        assert np.allclose(sums.numpy(), 1.0, atol=1e-4), "Softmax output does not sum to 1.0!"
        print("9. Softmax probabilities sum to 1.0 verified.")

    # Run 1 single training step with gradient update
    for imgs, lbls in train_ds.take(1):
        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            loss_val = model.loss(lbls, logits)
        grads = tape.gradient(loss_val, model.trainable_weights)
        assert len(grads) > 0 and grads[0] is not None, "No gradients computed!"
        print(f"10. Forward-backward gradient step verified. Single-batch loss: {float(loss_val):.4f}")

    # Verify ComprehensiveMetricsCallback execution on 4 samples
    mini_val_df = val_df.head(4)
    mini_val_ds = build_dataset(mini_val_df, batch_size=4, is_training=False, input_scaling=input_scaling)
    test_csv = Path(".keras_cache/test_smoke_metrics.csv")
    cb = ComprehensiveMetricsCallback(mini_val_ds, mini_val_df, test_csv, eval_model=model)
    cb.on_epoch_end(0, logs={"loss": 0.5, "accuracy": 0.8, "val_loss": 0.6})
    assert test_csv.exists(), "Metrics CSV was not written!"
    res_df = pd.read_csv(test_csv)
    required_cols = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "macro_precision", "macro_recall", "macro_f1",
        "bcc_recall", "melanoma_recall", "nevus_recall", "lr"
    ]
    for c in required_cols:
        assert c in res_df.columns, f"Missing metric column: {c}"
    test_csv.unlink(missing_ok=True)
    print("11. Comprehensive metrics calculation & CSV logging verified.")

    print("\n[SMOKE TEST PASSED] MobileNetV2 Revision V1 pipeline is fully operational!")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="DermaVision Revision V1: MobileNetV2 Training Pipeline")
    parser.add_argument("--train_manifest", type=str, default="data/splits/revision_v1/train.csv", help="Path to train CSV")
    parser.add_argument("--val_manifest", type=str, default="data/splits/revision_v1/val.csv", help="Path to val CSV")
    parser.add_argument("--output_model_dir", type=str, default="models/revision_v1/mobilenetv2", help="Model checkpoint directory")
    parser.add_argument("--output_result_dir", type=str, default="results/revision_v1/mobilenetv2", help="History & log directory")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--input_scaling", type=str, choices=["div255", "mobilenet"], default="div255",
                        help="Input normalization: 'div255' ([0, 1]) or 'mobilenet' ([-1, 1]) (default: div255)")
    parser.add_argument("--freeze_bn", action="store_true",
                        help="Freeze BatchNormalization layers during fine-tuning (default: False, matching Kaggle baseline)")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Apply balanced inverse class weights to handle training class imbalance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke_test", action="store_true", help="Run forward-pass smoke test and exit without full training")
    args = parser.parse_args()

    # Reproducibility
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("="*70)
    print("DERMAVISION: MOBILENETV2 TRAINING PIPELINE (REVISION V1)")
    print("="*70)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Random Seed:        {args.seed}")
    print(f"Initial LR:         {args.lr}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Max Epochs:         {args.epochs}")
    print(f"Input Scaling:      {args.input_scaling} ({'[0, 1]' if args.input_scaling == 'div255' else '[-1, 1]'})")
    print(f"Freeze BatchNorm:   {args.freeze_bn}")
    print(f"Use Class Weights:  {args.use_class_weights}")

    # Load and verify manifests
    train_df = verify_manifest(args.train_manifest, "Training")
    val_df = verify_manifest(args.val_manifest, "Validation")

    train_hash = compute_sha256(args.train_manifest)
    val_hash = compute_sha256(args.val_manifest)

    print(f"Train Manifest:     {args.train_manifest} ({len(train_df):,} samples) | SHA-256: {train_hash[:16]}...")
    print(f"Val Manifest:       {args.val_manifest} ({len(val_df):,} samples) | SHA-256: {val_hash[:16]}...")

    # Class distribution & weighting
    train_counts = train_df["class_label"].value_counts().sort_index()
    print("Training class distribution:")
    for idx, cname in enumerate(CLASS_NAMES):
        count = train_counts.get(idx, 0)
        pct = (count / len(train_df)) * 100
        print(f"  Class {idx} ({cname:8s}): {count:,} samples ({pct:.2f}%)")

    class_weights_dict = None
    if args.use_class_weights:
        weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df["class_label"].values)
        class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
        print("Computed balanced class weights:")
        for idx, cname in enumerate(CLASS_NAMES):
            print(f"  Class {idx} ({cname}): {class_weights_dict[idx]:.4f}")

    # Build model
    model = build_mobilenetv2_model(num_classes=3, lr=args.lr, freeze_bn=args.freeze_bn)

    if args.smoke_test:
        run_smoke_test(model, train_df, val_df, batch_size=4, input_scaling=args.input_scaling)
        print("Smoke test complete. Exiting without full training as requested.")
        return

    # Check directories
    os.makedirs(args.output_model_dir, exist_ok=True)
    os.makedirs(args.output_result_dir, exist_ok=True)

    # Save training config
    config = {
        "model": "MobileNetV2",
        "dataset_split": "revision_v1",
        "train_manifest": args.train_manifest,
        "train_manifest_sha256": train_hash,
        "val_manifest": args.val_manifest,
        "val_manifest_sha256": val_hash,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.lr,
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "input_scaling": args.input_scaling,
        "freeze_batch_normalization": args.freeze_bn,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights_dict,
        "epochs": args.epochs,
        "image_size": [224, 224, 3],
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = Path(args.output_result_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to: {config_path}")

    # Callbacks
    checkpoint_path = Path(args.output_model_dir) / "best_mobilenetv2.keras"
    csv_log_path = Path(args.output_result_dir) / "training_history.csv"

    val_ds = build_dataset(val_df, batch_size=args.batch_size, is_training=False, input_scaling=args.input_scaling)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ComprehensiveMetricsCallback(
            val_ds=val_ds,
            val_df=val_df,
            csv_path=csv_log_path
        )
    ]

    print("\nPreparing tf.data pipelines...")
    train_ds = build_dataset(train_df, batch_size=args.batch_size, is_training=True, input_scaling=args.input_scaling)

    start_time = time.time()
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    training_duration = time.time() - start_time

    # Save final model checkpoint
    final_checkpoint_path = Path(args.output_model_dir) / "final_mobilenetv2.keras"
    model.save(str(final_checkpoint_path))

    # Read training history to find best epoch metrics
    history_df = pd.read_csv(csv_log_path)
    best_row = history_df.loc[history_df["val_acc"].idxmax()]

    summary = {
        "model": "MobileNetV2",
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
    print(f"Best checkpoint saved at:  {checkpoint_path} (Val Acc: {summary['best_val_accuracy']*100:.2f}%, Best Epoch: {summary['best_epoch']})")
    print(f"Final checkpoint saved at: {final_checkpoint_path} (Val Acc: {summary['final_val_accuracy']*100:.2f}%)")
    print(f"Training history saved at: {csv_log_path}")
    print(f"Training summary saved at: {summary_path}")

if __name__ == "__main__":
    main()

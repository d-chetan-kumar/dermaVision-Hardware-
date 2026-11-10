"""
prune_and_train_mobilenetv2.py
================================================================================
DermaVision: MobileNetV2 Gradual Unstructured Weight Pruning Pipeline
--------------------------------------------------------------------------------
1. Loads original checkpoint: 'models/best_mobilenetv2.keras'.
2. Transfers weights into tf_keras model (Dense(256) -> Dense(64) -> Dense(3)).
3. Selectively wraps only supported weight-bearing layers (Conv2D and Dense)
   with tfmot.sparsity.keras.prune_low_magnitude using PolynomialDecay:
   - initial_sparsity: 0.0
   - final_sparsity: 0.50
   - begin_step: 0
   - end_step: 1560 (312 batches * 5 epochs)
   - frequency: 100
   Leaves Input, Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU, Add
   unwrapped and untouched.
4. Prepares original 9,956 train / 2,489 val split with / 255.0 scaling and class
   mapping {0: BCC, 1: Melanoma, 2: Nevus}.
5. Fine-tunes for 5 epochs with Adam(learning_rate=1e-5), sparse categorical
   crossentropy, batch_size=32, UpdatePruningStep callback, and per-epoch audits.
6. Saves per-epoch checkpoints in 'results/pruning/checkpoints/'.
7. Saves training history CSV to 'results/pruning/mobilenetv2_pruning_history.csv'.
8. Strips pruning wrappers with tfmot.sparsity.keras.strip_pruning() from the best
   post-pruning checkpoint (sparsity >= 49.5%).
9. Transfers stripped sparse weights into Keras 3 format and saves as
   'models/best_mobilenetv2_pruned.keras'.
================================================================================
"""

import os
import sys
import time
from pathlib import Path

# Ensure writable cache directory and enable oneDNN optimizations for fast CPU execution
os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")
os.environ["MPLCONFIGDIR"] = os.path.abspath(".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude

from evaluation import prepare_validation_dataset, build_image_path

# Configuration Constants
ORIG_MODEL_PATH = "models/best_mobilenetv2.keras"
OUTPUT_PRUNED_PATH = "models/best_mobilenetv2_pruned.keras"
HISTORY_CSV_PATH = "results/pruning/mobilenetv2_pruning_history.csv"
CHECKPOINTS_DIR = "results/pruning/checkpoints"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
INITIAL_SPARSITY = 0.0
FINAL_SPARSITY = 0.50
PRUNING_FREQUENCY = 100
LEARNING_RATE = 1e-5
SEED = 42

def count_layer_sparsity(model):
    """
    Computes exact element-wise zero count and sparsity percentage
    across Conv2D and Dense kernels.
    """
    conv_dense_zeros, conv_dense_total = 0, 0
    all_zeros, all_total = 0, 0
    layer_details = []

    for layer in model.layers:
        t = type(layer).__name__
        if isinstance(layer, PruneLowMagnitude):
            inner_layer = layer.layer
            inner_t = type(inner_layer).__name__
            w = inner_layer.kernel.numpy()
            zeros = int(np.sum(w == 0))
            tot = int(w.size)
            conv_dense_zeros += zeros
            conv_dense_total += tot
            all_zeros += zeros
            all_total += tot
            layer_details.append({
                "layer_name": inner_layer.name,
                "type": inner_t,
                "zeros": zeros,
                "total": tot,
                "sparsity": zeros / tot if tot else 0.0
            })
            # Also count bias if present
            if inner_layer.bias is not None:
                b = inner_layer.bias.numpy()
                all_zeros += int(np.sum(b == 0))
                all_total += int(b.size)
        else:
            # Unwrapped layers (e.g. DepthwiseConv2D, BatchNorm)
            for weight in layer.weights:
                w = weight.numpy()
                all_zeros += int(np.sum(w == 0))
                all_total += int(w.size)

    prunable_sparsity = conv_dense_zeros / conv_dense_total if conv_dense_total else 0.0
    overall_sparsity = all_zeros / all_total if all_total else 0.0
    return {
        "prunable_zeros": conv_dense_zeros,
        "prunable_total": conv_dense_total,
        "prunable_sparsity": prunable_sparsity,
        "all_zeros": all_zeros,
        "all_total": all_total,
        "overall_sparsity": overall_sparsity,
        "layer_details": layer_details
    }


class PruningAuditCallback(tf_keras.callbacks.Callback):
    """
    Custom callback to log exact train/val metrics and kernel sparsity
    at the end of each epoch, and save per-epoch checkpoint weights.
    """
    def __init__(self, val_dataset, history_log, checkpoints_dir):
        super().__init__()
        self.val_dataset = val_dataset
        self.history_log = history_log
        self.checkpoints_dir = checkpoints_dir
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} Starting ---")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        sparsity_stats = count_layer_sparsity(self.model)
        prunable_sparsity = sparsity_stats["prunable_sparsity"] * 100.0
        overall_sparsity = sparsity_stats["overall_sparsity"] * 100.0

        # Save per-epoch model
        ckpt_path = os.path.join(self.checkpoints_dir, f"mobilenetv2_pruned_epoch_{epoch+1:02d}.h5")
        self.model.save(ckpt_path)

        train_loss = float(logs.get("loss", 0.0))
        train_acc = float(logs.get("accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "prunable_sparsity_percent": prunable_sparsity,
            "overall_sparsity_percent": overall_sparsity,
            "prunable_zeros": sparsity_stats["prunable_zeros"],
            "prunable_total": sparsity_stats["prunable_total"],
            "epoch_duration_seconds": elapsed,
            "checkpoint_path": ckpt_path
        }
        self.history_log.append(record)

        print(f"Epoch {epoch+1}/{EPOCHS} Complete ({elapsed:.1f}s):")
        print(f"  Train Acc: {train_acc*100:6.2f}% | Train Loss: {train_loss:.4f}")
        print(f"  Val Acc:   {val_acc*100:6.2f}% | Val Loss:   {val_loss:.4f}")
        print(f"  Prunable Kernel Sparsity: {prunable_sparsity:5.2f}% ({sparsity_stats['prunable_zeros']:,} / {sparsity_stats['prunable_total']:,} zeros)")
        print(f"  Overall Total Sparsity:   {overall_sparsity:5.2f}%")
        print(f"  Saved Checkpoint: {ckpt_path}")


def main():
    print("=" * 80)
    print("DERMAVISION: MOBILENETV2 GRADUAL UNSTRUCTURED WEIGHT PRUNING")
    print("=" * 80)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs("results/pruning", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------------------------------------
    # Step 1: Reconstruct Original Dataset Partitions
    # --------------------------------------------------------------------------
    print("\n[Step 1/6] Loading original dataset and preparing partitions...")
    img_dir = "ISIC_2019_Training_Input/ISIC_2019_Training_Input" if os.path.exists("ISIC_2019_Training_Input/ISIC_2019_Training_Input") else "ISIC_2019_Training_Input"
    train_df, val_df = prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", img_dir)
    train_df["image_path"] = train_df["image"].apply(lambda x: str(build_image_path(x, Path(img_dir))))

    assert len(train_df) == 9956, f"Expected 9,956 training samples, got {len(train_df)}"
    assert len(val_df) == 2489, f"Expected 2,489 validation samples, got {len(val_df)}"
    print(f"  Training set verified:   {len(train_df):,} images")
    print(f"  Validation set verified: {len(val_df):,} images")

    # TensorFlow tf.data Pipeline
    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0  # Exact original preprocessing
        return img, label

    # Full training dataset with global random shuffle on paths
    train_paths = train_df["image_path"].values
    train_labels = train_df["label"].values.astype(np.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(train_df), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Deterministic validation dataset
    val_paths = val_df["image_path"].values
    val_labels = val_df["label"].values.astype(np.int32)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
    total_steps = steps_per_epoch * EPOCHS
    print(f"  Steps per epoch: {steps_per_epoch} | Total pruning steps: {total_steps}")

    # --------------------------------------------------------------------------
    # Step 2: Load Original Checkpoint & Build tf_keras Architecture
    # --------------------------------------------------------------------------
    print(f"\n[Step 2/6] Loading source model from '{ORIG_MODEL_PATH}'...")
    orig_k3 = keras.models.load_model(ORIG_MODEL_PATH)
    orig_k3_bb = orig_k3.get_layer("mobilenetv2_1.00_224")

    print("  Constructing exact tf_keras functional architecture...")
    base_bb = tf_keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    x = base_bb.output
    x = tf_keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf_keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf_keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf_keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = tf_keras.layers.Dropout(0.1, name="dropout_1")(x)
    out = tf_keras.layers.Dense(3, activation="softmax", name="dense_2")(x)
    tfk_model = tf_keras.Model(inputs=base_bb.input, outputs=out)

    print("  Transferring pretrained weights layer-by-layer...")
    for l in orig_k3_bb.layers:
        if l.name != "input_layer":
            base_bb.get_layer(l.name).set_weights(l.get_weights())
    for name in ["dense", "dense_1", "dense_2"]:
        tfk_model.get_layer(name).set_weights(orig_k3.get_layer(name).get_weights())

    # Parity verification
    dummy = np.random.rand(2, 224, 224, 3).astype("float32")
    p_orig = orig_k3(dummy, training=False).numpy()
    p_tfk = tfk_model(dummy, training=False).numpy()
    max_parity_diff = float(np.max(np.abs(p_orig - p_tfk)))
    print(f"  Weight transfer parity check: max diff = {max_parity_diff:.2e}")
    assert max_parity_diff < 1e-5, f"Parity check failed: {max_parity_diff}"

    # --------------------------------------------------------------------------
    # Step 3: Configure Gradual Pruning with PolynomialDecay
    # --------------------------------------------------------------------------
    print("\n[Step 3/6] Applying selective layer pruning with PolynomialDecay...")
    prune_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=INITIAL_SPARSITY,
        final_sparsity=FINAL_SPARSITY,
        begin_step=0,
        end_step=total_steps,
        frequency=PRUNING_FREQUENCY
    )

    def apply_selective_pruning(layer):
        # Prune only supported weight-bearing layers (Conv2D and Dense)
        if isinstance(layer, (tf_keras.layers.Conv2D, tf_keras.layers.Dense)):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=prune_schedule)
        return layer

    pruned_model = tf_keras.models.clone_model(tfk_model, clone_function=apply_selective_pruning)

    # Audit wrapped layers
    wrapped_count = sum(1 for l in pruned_model.layers if isinstance(l, PruneLowMagnitude))
    unwrapped_count = len(pruned_model.layers) - wrapped_count
    print(f"  Total layers:     {len(pruned_model.layers)}")
    print(f"  Wrapped layers:   {wrapped_count} (35 Conv2D + 3 Dense kernels)")
    print(f"  Preserved layers: {unwrapped_count} (Dropout, BatchNorm, Pooling, etc. completely untouched)")
    assert wrapped_count == 38, f"Expected 38 wrapped layers, got {wrapped_count}"

    pruned_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------------------------
    # Step 4: Fine-Tuning for 5 Epochs with UpdatePruningStep
    # --------------------------------------------------------------------------
    print("\n[Step 4/6] Fine-tuning pruned model for 5 epochs on CPU...")
    history_log = []
    audit_cb = PruningAuditCallback(val_dataset=val_ds, history_log=history_log, checkpoints_dir=CHECKPOINTS_DIR)
    update_cb = tfmot.sparsity.keras.UpdatePruningStep()

    t_train_start = time.time()
    pruned_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[update_cb, audit_cb],
        verbose=1
    )
    total_train_duration = time.time() - t_train_start
    print(f"\nFine-tuning completed in {total_train_duration/60:.2f} minutes.")

    # Save training history CSV
    history_df = pd.DataFrame(history_log)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"  Training history saved to: '{HISTORY_CSV_PATH}'")

    # --------------------------------------------------------------------------
    # Step 5: Checkpoint Selection & Wrapper Stripping
    # --------------------------------------------------------------------------
    print("\n[Step 5/6] Inspecting epoch history & selecting best target-sparsity checkpoint...")
    print(history_df[["epoch", "train_accuracy", "val_accuracy", "val_loss", "prunable_sparsity_percent"]].to_string(index=False))

    # Target sparsity criterion: select checkpoint with pruning >= 49.0% having highest val_accuracy
    target_sparsity_candidates = [r for r in history_log if r["prunable_sparsity_percent"] >= 49.0]
    if target_sparsity_candidates:
        best_record = max(target_sparsity_candidates, key=lambda r: r["val_accuracy"])
        print(f"\n  Selected Best Target-Sparsity Checkpoint from Epoch {best_record['epoch']}:")
        print(f"    Val Accuracy: {best_record['val_accuracy']*100:.2f}%")
        print(f"    Val Loss:     {best_record['val_loss']:.4f}")
        print(f"    Sparsity:     {best_record['prunable_sparsity_percent']:.2f}%")
    else:
        # Fallback to final epoch
        best_record = history_log[-1]
        print(f"\n  Selected Final Epoch Checkpoint (Epoch {best_record['epoch']}):")

    best_ckpt_path = best_record["checkpoint_path"]
    print(f"  Loading best checkpoint from '{best_ckpt_path}'...")
    with tfmot.sparsity.keras.prune_scope():
        best_pruned_wrapper = tf_keras.models.load_model(best_ckpt_path)

    print("  Applying tfmot.sparsity.keras.strip_pruning()...")
    stripped_model = tfmot.sparsity.keras.strip_pruning(best_pruned_wrapper)

    # --------------------------------------------------------------------------
    # Step 6: Transfer to Native Keras 3 Model & Verify Sparsity Parity
    # --------------------------------------------------------------------------
    print("\n[Step 6/6] Converting stripped model into native Keras 3 .keras format...")
    k3_pruned_clone = keras.models.clone_model(orig_k3)
    k3_pruned_bb = k3_pruned_clone.get_layer("mobilenetv2_1.00_224")
    bb_names = set(l.name for l in k3_pruned_bb.layers)

    for l in stripped_model.layers:
        if l.name in bb_names:
            k3_pruned_bb.get_layer(l.name).set_weights(l.get_weights())
        elif l.name in ["dense", "dense_1", "dense_2"]:
            k3_pruned_clone.get_layer(l.name).set_weights(l.get_weights())

    # Save final stripped model
    k3_pruned_clone.save(OUTPUT_PRUNED_PATH)
    print(f"  Saved final stripped pruned model: '{OUTPUT_PRUNED_PATH}' ({os.path.getsize(OUTPUT_PRUNED_PATH)/(1024*1024):.2f} MB)")

    # Verify reloading via standard Keras
    print("  Verifying reload from disk...")
    reloaded_pruned = keras.models.load_model(OUTPUT_PRUNED_PATH)
    p_strip = stripped_model(dummy, training=False).numpy()
    p_reload = reloaded_pruned(dummy, training=False).numpy()
    strip_reload_diff = float(np.max(np.abs(p_strip - p_reload)))
    print(f"  Stripped vs Reloaded Keras 3 parity check: max diff = {strip_reload_diff:.2e}")
    assert strip_reload_diff < 1e-5, f"Reload parity mismatch: {strip_reload_diff}"

    # Also save separate designation for best overall validation checkpoint
    best_overall = max(history_log, key=lambda r: r["val_accuracy"])
    print(f"  Best overall validation epoch: Epoch {best_overall['epoch']} (Val Acc: {best_overall['val_accuracy']*100:.2f}%, Sparsity: {best_overall['prunable_sparsity_percent']:.2f}%)")
    if best_overall["epoch"] != best_record["epoch"]:
        print(f"  Note: Best target-sparsity (>=49%) epoch was Epoch {best_record['epoch']} (Val Acc: {best_record['val_accuracy']*100:.2f}%, Sparsity: {best_record['prunable_sparsity_percent']:.2f}%)")

    # Element-wise sparsity audit on reloaded model
    c_zeros, c_total = 0, 0
    d_zeros, d_total = 0, 0
    all_zeros, all_total = 0, 0

    reloaded_bb = reloaded_pruned.get_layer("mobilenetv2_1.00_224")
    for l in reloaded_bb.layers:
        if type(l).__name__ == "Conv2D":
            w = l.kernel.numpy()
            z = int(np.sum(w == 0))
            tot = int(w.size)
            c_zeros += z
            c_total += tot
            all_zeros += z
            all_total += tot
        else:
            for weight in l.weights:
                w = weight.numpy()
                all_zeros += int(np.sum(w == 0))
                all_total += int(w.size)

    for name in ["dense", "dense_1", "dense_2"]:
        l = reloaded_pruned.get_layer(name)
        w = l.kernel.numpy()
        z = int(np.sum(w == 0))
        tot = int(w.size)
        d_zeros += z
        d_total += tot
        all_zeros += z
        all_total += tot
        if l.bias is not None:
            b = l.bias.numpy()
            all_zeros += int(np.sum(b == 0))
            all_total += int(b.size)

    prunable_zeros = c_zeros + d_zeros
    prunable_total = c_total + d_total
    actual_prunable_sparsity = prunable_zeros / prunable_total * 100.0
    actual_overall_sparsity = all_zeros / all_total * 100.0

    print("\n" + "=" * 80)
    print("FINAL MEASURED SPARSITY AUDIT")
    print("=" * 80)
    print(f"Conv2D Kernels:           {c_zeros:,} / {c_total:,} zeros ({c_zeros/c_total*100:.2f}%)")
    print(f"Dense Kernels:            {d_zeros:,} / {d_total:,} zeros ({d_zeros/d_total*100:.2f}%)")
    print(f"Prunable Weights Total:   {prunable_zeros:,} / {prunable_total:,} zeros ({actual_prunable_sparsity:.2f}%)")
    print(f"Total Model Parameters:   {all_zeros:,} / {all_total:,} zeros ({actual_overall_sparsity:.2f}%)")
    print("=" * 80)
    print("TRAINING AND PRUNING COMPLETE! PROCEED TO EVALUATION.")

if __name__ == "__main__":
    main()

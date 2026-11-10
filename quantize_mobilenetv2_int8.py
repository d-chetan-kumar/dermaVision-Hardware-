"""
quantize_mobilenetv2_int8.py
================================================================================
DermaVision: Full Integer INT8 Quantization of Original MobileNetV2
--------------------------------------------------------------------------------
Converts 'models/best_mobilenetv2.keras' into 'models/best_mobilenetv2_full_int8.tflite'
using TensorFlow Lite Converter with:
- Optimizations: [tf.lite.Optimize.DEFAULT]
- Representative calibration dataset: 250 images sampled deterministically
  from the ORIGINAL TRAINING SET ONLY (zero validation or test leakage).
- Supported ops: [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
- Inference input type: tf.int8
- Inference output type: tf.int8
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure writable cache directory for Keras & Matplotlib
os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")
os.environ["MPLCONFIGDIR"] = os.path.abspath(".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from evaluation import prepare_validation_dataset

KERAS_MODEL_PATH = "models/best_mobilenetv2.keras"
OUTPUT_INT8_PATH = "models/best_mobilenetv2_full_int8.tflite"
IMG_SIZE = (224, 224)
CALIBRATION_SAMPLES = 250
SEED = 42

def main():
    print("=" * 70)
    print("DERMAVISION: FULL INTEGER INT8 QUANTIZATION")
    print("=" * 70)

    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(f"Original Keras model not found at: {KERAS_MODEL_PATH}")

    # 1. Load original dataset partitions to extract training samples strictly
    print("\n[Step 1/5] Reconstructing original dataset partitions...")
    img_dir = "ISIC_2019_Training_Input/ISIC_2019_Training_Input" if os.path.exists("ISIC_2019_Training_Input/ISIC_2019_Training_Input") else "ISIC_2019_Training_Input"
    train_df, val_df = prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", img_dir)
    print(f"  Total training images:   {len(train_df):,}")
    print(f"  Total validation images: {len(val_df):,}")

    # Sample calibration data exclusively from train_df
    calib_df = train_df.sample(n=CALIBRATION_SAMPLES, random_state=SEED).reset_index(drop=True)
    print(f"  Selected {len(calib_df)} calibration images strictly from training set.")
    print("  Calibration class distribution:")
    for cls_name, count in calib_df["class"].value_counts().items():
        print(f"    - {cls_name}: {count} ({count/len(calib_df)*100:.1f}%)")

    # 2. Load original Keras model
    print(f"\n[Step 2/5] Loading original Keras model from '{KERAS_MODEL_PATH}'...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print(f"  Model input shape:  {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")
    print(f"  Total parameters:   {model.count_params():,}")

    # 3. Create representative dataset generator
    print("\n[Step 3/5] Defining representative dataset generator...")
    def representative_dataset_gen():
        for idx, row in calib_df.iterrows():
            img_path = row["image_path"]
            raw = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0  # Exact original preprocessing
            img = tf.expand_dims(img, axis=0)      # Shape: (1, 224, 224, 3)
            yield [img]

    # 4. Configure Full Integer INT8 Converter
    print("\n[Step 4/5] Configuring TensorFlow Lite Converter for Full INT8 Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("  Starting quantization conversion (calibrating activation ranges)...")
    t0 = time.time()
    tflite_quant_model = converter.convert()
    conv_time = time.time() - t0
    print(f"  Conversion completed successfully in {conv_time:.2f} seconds.")

    # 5. Save quantized model
    print(f"\n[Step 5/5] Saving Full INT8 model to '{OUTPUT_INT8_PATH}'...")
    with open(OUTPUT_INT8_PATH, "wb") as f:
        f.write(tflite_quant_model)

    file_size_bytes = os.path.getsize(OUTPUT_INT8_PATH)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  Saved: {OUTPUT_INT8_PATH}")
    print(f"  File size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

    # 6. Verify interpreter and tensor parameters
    print("\n" + "=" * 70)
    print("VERIFYING QUANTIZATION TENSOR PARAMETERS")
    print("=" * 70)
    interpreter = tf.lite.Interpreter(model_path=OUTPUT_INT8_PATH)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    print("Input Tensor Details:")
    print(f"  Name:        {in_details['name']}")
    print(f"  Shape:       {in_details['shape']}")
    print(f"  Dtype:       {in_details['dtype']}")
    print(f"  Scale:       {in_details['quantization_parameters']['scales'][0]}")
    print(f"  Zero Point:  {in_details['quantization_parameters']['zero_points'][0]}")

    print("\nOutput Tensor Details:")
    print(f"  Name:        {out_details['name']}")
    print(f"  Shape:       {out_details['shape']}")
    print(f"  Dtype:       {out_details['dtype']}")
    print(f"  Scale:       {out_details['quantization_parameters']['scales'][0]}")
    print(f"  Zero Point:  {out_details['quantization_parameters']['zero_points'][0]}")

    assert in_details['dtype'] == np.int8, f"Expected input dtype int8, got {in_details['dtype']}"
    assert out_details['dtype'] == np.int8, f"Expected output dtype int8, got {out_details['dtype']}"
    print("\nSUCCESS: Input and Output tensors are confirmed INT8!")

    # Test single image inference
    test_img = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    in_scale = in_details['quantization_parameters']['scales'][0]
    in_zp = in_details['quantization_parameters']['zero_points'][0]
    out_scale = out_details['quantization_parameters']['scales'][0]
    out_zp = out_details['quantization_parameters']['zero_points'][0]

    # Quantize input
    q_in = np.round(test_img / in_scale + in_zp)
    q_in = np.clip(q_in, -128, 127).astype(np.int8)

    interpreter.set_tensor(in_details['index'], q_in)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_details['index'])

    # Dequantize output
    dequant_probs = (q_out.astype(np.float32) - out_zp) * out_scale
    print(f"\nDummy Test Inference Output (Raw INT8): {q_out}")
    print(f"Dequantized Output Probabilities:       {dequant_probs}")
    print(f"Sum of Dequantized Probabilities:       {float(np.sum(dequant_probs)):.4f}")

if __name__ == "__main__":
    main()


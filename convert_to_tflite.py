"""
convert_to_tflite.py

Converts best_mobilenetv2.keras to a quantized TFLite model
for Raspberry Pi (ARM64) deployment.

Run this ONCE on your Windows/Linux PC (where TensorFlow is installed):
    python convert_to_tflite.py

The output `models/best_mobilenetv2.tflite` (~8 MB) can then be committed
to GitHub and loaded on the Pi using tflite-runtime.
"""

import os
import tensorflow as tf

KERAS_MODEL_PATH = "models/best_mobilenetv2.keras"
TFLITE_MODEL_PATH = "models/best_mobilenetv2.tflite"

def convert():
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{KERAS_MODEL_PATH}'.\n"
            "Place best_mobilenetv2.keras inside the models/ folder first."
        )

    print(f"[1/3] Loading Keras model from: {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print(f"      Input shape: {model.input_shape}")
    print(f"      Output shape: {model.output_shape}")

    print("[2/3] Converting to TFLite with dynamic range quantization ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic range quantization: ~4x smaller, 2-3x faster on CPU, no data needed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs("models", exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"[3/3] Saved TFLite model to: {TFLITE_MODEL_PATH} ({size_mb:.2f} MB)")
    print("\n✅ Conversion complete! Commit models/best_mobilenetv2.tflite to GitHub.")

if __name__ == "__main__":
    convert()

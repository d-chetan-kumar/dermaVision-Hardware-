"""
prediction_int8.py
================================================================================
DermaVision: Standalone Full Integer INT8 TFLite Inference Module
--------------------------------------------------------------------------------
Provides quantized inference using 'models/best_mobilenetv2_full_int8.tflite'.
Correctly handles:
- Input scale and zero-point quantization: round(fp32 / scale + zp) clipped to [-128, 127]
- Output scale and zero-point dequantization: (int8 - zp) * scale
- Probabilities mapped to classes: BCC (0), Melanoma (1), Nevus (2)
================================================================================
"""

import os
import json
import numpy as np
from PIL import Image

# Ensure writable cache directory for Keras & Matplotlib
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.abspath(".keras_cache")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Use LiteRT (ai_edge_litert), tflite_runtime, or tensorflow.lite
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

MODEL_PATH = "models/best_mobilenetv2_full_int8.tflite"
CLASS_MAPPING = {0: "BCC", 1: "Melanoma", 2: "Nevus"}
IMG_SIZE = (224, 224)

class MobileNetV2FullINT8Predictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"INT8 model not found at '{model_path}'.")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # Extract quantization parameters
        self.in_scale = float(self.input_details['quantization_parameters']['scales'][0])
        self.in_zero_point = int(self.input_details['quantization_parameters']['zero_points'][0])
        self.out_scale = float(self.output_details['quantization_parameters']['scales'][0])
        self.out_zero_point = int(self.output_details['quantization_parameters']['zero_points'][0])

        assert self.input_details['dtype'] == np.int8, "Input tensor must be INT8"
        assert self.output_details['dtype'] == np.int8, "Output tensor must be INT8"

    def preprocess(self, image_input):
        """Preprocesses image: RGB conversion, resize to 224x224, divide by 255.0 to [0, 1]."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.open(image_input).convert("RGB")

        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
        return img_batch

    def quantize_input(self, img_batch_fp32):
        """Quantizes normalized FP32 input into signed INT8 in [-128, 127]."""
        q_in = np.round(img_batch_fp32 / self.in_scale + self.in_zero_point)
        q_in = np.clip(q_in, -128, 127).astype(np.int8)
        return q_in

    def dequantize_output(self, q_output):
        """Dequantizes INT8 output into FP32 probabilities."""
        dequant = (q_output.astype(np.float32) - self.out_zero_point) * self.out_scale
        # Guard for softmax normalization
        s = np.sum(dequant)
        if s > 0:
            dequant = dequant / s
        return dequant

    def predict(self, image_input):
        """Runs full integer INT8 inference on a single image."""
        img_batch_fp32 = self.preprocess(image_input)
        q_in = self.quantize_input(img_batch_fp32)

        self.interpreter.set_tensor(self.input_details['index'], q_in)
        self.interpreter.invoke()
        q_out = self.interpreter.get_tensor(self.output_details['index'])[0]

        probabilities = self.dequantize_output(q_out)
        pred_idx = int(np.argmax(probabilities))
        pred_class = CLASS_MAPPING.get(pred_idx, f"Class {pred_idx}")
        confidence = float(probabilities[pred_idx]) * 100.0

        probs_dict = {
            CLASS_MAPPING[i]: round(float(prob) * 100.0, 2)
            for i, prob in enumerate(probabilities)
        }

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence, 2),
            "probabilities": probs_dict,
            "raw_quantized_output": q_out.tolist(),
            "quantization_params": {
                "input_scale": self.in_scale,
                "input_zero_point": self.in_zero_point,
                "output_scale": self.out_scale,
                "output_zero_point": self.out_zero_point
            }
        }

def test_inference():
    predictor = MobileNetV2FullINT8Predictor()
    dummy = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    res = predictor.predict(dummy)
    print("Self-test completed successfully:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test_inference()

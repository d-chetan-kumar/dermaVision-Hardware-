import os
import numpy as np

# Use tflite-runtime on ARM (Raspberry Pi) or fall back to tensorflow.lite on desktop
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from preprocessing import preprocess_image
from validator import is_valid_dermatology_image
from utils import load_class_mapping

# Constants
MODEL_PATH = "models/best_mobilenetv2.tflite"

def load_prediction_model():
    """
    Loads the TFLite model from the specified path and allocates tensors.
    Returns the initialized Interpreter object.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Run convert_to_tflite.py on a machine with TensorFlow installed first."
        )
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def _run_inference(interpreter, image_batch):
    """
    Runs TFLite inference on a preprocessed image batch (shape: 1, 224, 224, 3).
    Returns a 1D numpy array of class probabilities.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_batch)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions

def predict_image(image, model=None, classes=None):
    """
    Predicts the disease using the TFLite model.
    1. Validate/prepare image
    2. Call preprocessing.py
    3. Run TFLite inference
    4. Calculate max index and map to class
    5. Return structured result
    """
    if not is_valid_dermatology_image(image):
        raise ValueError("Invalid dermatology image. Please provide a clear, valid image.")

    if model is None:
        model = load_prediction_model()

    if classes is None:
        classes = load_class_mapping()

    image_batch, _ = preprocess_image(image)

    predictions = _run_inference(model, image_batch)
    predicted_index = int(np.argmax(predictions))

    predicted_class = classes[predicted_index] if predicted_index < len(classes) else f"Class {predicted_index}"
    confidence = float(predictions[predicted_index]) * 100

    probabilities = {}
    for i, prob in enumerate(predictions):
        cls_name = classes[i] if i < len(classes) else f"Class {i}"
        probabilities[cls_name] = float(prob) * 100

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }

def predict_top_classes(model, img_batch, class_names, top_k=5):
    """
    Predicts top k classes for an image batch and returns a sorted list of (class_name, probability).
    """
    predictions = _run_inference(model, img_batch)

    class_probs = []
    for i, prob in enumerate(predictions):
        cls_name = class_names[i] if i < len(class_names) else f"Class {i}"
        class_probs.append((cls_name, float(prob)))

    class_probs.sort(key=lambda x: x[1], reverse=True)
    return class_probs[:top_k]


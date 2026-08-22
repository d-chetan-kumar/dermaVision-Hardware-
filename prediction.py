import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

from preprocessing import preprocess_image
from validator import is_valid_dermatology_image
from utils import load_class_mapping

# Constants
MODEL_PATH = "models/best_mobilenetv2.keras"

def load_prediction_model():
    """
    Loads the compiled Keras model from the specified path.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    return keras_load_model(MODEL_PATH)

def predict_image(image, model=None, classes=None):
    """
    Predicts the disease using the model.
    1. Validate/prepare image
    2. Call preprocessing.py
    3. Run model.predict
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
    
    predictions = model.predict(image_batch, verbose=0)[0]
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
    predictions = model.predict(img_batch, verbose=0)[0]
    
    class_probs = []
    for i, prob in enumerate(predictions):
        cls_name = class_names[i] if i < len(class_names) else f"Class {i}"
        class_probs.append((cls_name, float(prob)))
        
    class_probs.sort(key=lambda x: x[1], reverse=True)
    return class_probs[:top_k]

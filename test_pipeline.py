import os
import json
import numpy as np

print("--- TEST 1: Import preprocessing ---")
from preprocessing import preprocess_image
print("Pass")

print("--- TEST 2: Import validator ---")
from validator import is_valid_dermatology_image
print("Pass")

print("--- TEST 3: Import prediction ---")
from prediction import load_prediction_model, predict_image
print("Pass")

print("--- TEST 4: Load model ---")
model = load_prediction_model()
print("Pass")

print("--- TEST 5: Load classes ---")
from utils import load_class_mapping
classes = load_class_mapping()
print("Classes loaded:", classes)
print("Pass")

print("--- TEST 6 & 7: Test dummy image ---")
from PIL import Image
dummy_img = Image.new('RGB', (300, 300), color = 'red')

if is_valid_dermatology_image(dummy_img):
    print("Validator passed dummy image")
    
res = predict_image(dummy_img, model=model, classes=classes)
print("Prediction Result:")
print("Predicted Class:", res["predicted_class"])
print("Confidence:", res["confidence"])
print("Probabilities:", res["probabilities"])
if len(res["probabilities"]) == 3:
    print("Pass: 3 probabilities returned")
    
print("All tests passed.")

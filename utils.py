import json
import os
import numpy as np
from PIL import Image

def load_class_mapping(file_path="models/class_names.json"):
    """
    Safely load the class names mapping from the JSON file.
    Expected format: {"0": "BCC", "1": "Melanoma", "2": "Nevus"}
    Returns a list: ["BCC", "Melanoma", "Nevus"] ordered by key.
    """
    try:
        if not os.path.exists(file_path):
            return ["BCC", "Melanoma", "Nevus"]
        with open(file_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
            # Ensure correct order 0, 1, 2
            classes = [class_mapping[str(i)] for i in range(len(class_mapping))]
            return classes
    except Exception:
        return ["BCC", "Melanoma", "Nevus"]

def format_probabilities(probabilities_dict):
    """
    Format the probabilities dictionary into a list of strings for UI display.
    """
    formatted = []
    for cls, prob in probabilities_dict.items():
        formatted.append(f"{cls}: {prob:.2f}%")
    return formatted

def evaluate_image_quality(img):
    """
    Computes contrast, brightness, focus/blur metrics, and details of the image.
    Generates recommendations for quality improvement.
    """
    img_gray = img.convert("L")
    img_arr = np.array(img_gray)
    
    # Calculate Mean Brightness (0 - 255)
    brightness = float(np.mean(img_arr))
    brightness_status = "PASS" if (45 <= brightness <= 215) else "CAUTION"
    brightness_text = "Optimal" if (45 <= brightness <= 215) else ("Too Dark" if brightness < 45 else "Too Bright")
    
    # Calculate Contrast (Standard Deviation)
    contrast = float(np.std(img_arr))
    contrast_status = "PASS" if contrast >= 25 else "CAUTION"
    contrast_text = "Optimal" if contrast >= 25 else "Low Contrast"
    
    # Calculate Sharpness/Focus Detail (variance of high frequency differences)
    diff_h = np.diff(img_arr, axis=0)
    diff_v = np.diff(img_arr, axis=1)
    detail_score = float(np.var(diff_h) + np.var(diff_v))
    
    focus_status = "PASS" if detail_score >= 12.0 else "CAUTION"
    focus_text = "Optimal Focus" if detail_score >= 12.0 else "Low Focus/Blurry"
    
    # Generate recommendations list
    recommendations = []
    if brightness_status == "CAUTION":
        recommendations.append("Better Lighting")
    if focus_status == "CAUTION" and detail_score < 6.0:
        recommendations.append("Capture Sharper Image")
    if focus_status == "CAUTION" and detail_score >= 6.0:
        recommendations.append("Move Camera Closer")
    if contrast_status == "CAUTION":
        recommendations.append("Reduce Background")
        
    overall_status = "PASS" if (brightness_status == "PASS" and contrast_status == "PASS" and focus_status == "PASS") else "CAUTION"
    
    return {
        "brightness": brightness,
        "brightness_status": brightness_status,
        "brightness_text": brightness_text,
        "contrast": contrast,
        "contrast_status": contrast_status,
        "contrast_text": contrast_text,
        "focus_score": detail_score,
        "focus_status": focus_status,
        "focus_text": focus_text,
        "recommendations": recommendations,
        "overall_status": overall_status
    }

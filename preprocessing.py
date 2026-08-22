import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Validates, loads, resizes, and preprocesses an uploaded image for MobileNetV2.
    Preprocessing exactly matches Kaggle training:
    - RGB conversion
    - Resize to 224 x 224
    - Convert to float32
    - Normalize using / 255.0
    - Add batch dimension
    """
    # Handle different input types (uploaded file, path, or already opened PIL image)
    if hasattr(image, "path"):
        img = Image.open(image.path).convert('RGB')
    elif isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        img = Image.open(image).convert('RGB')
        
    img_resized = img.resize((224, 224))
    
    # EXACTLY AS REQUESTED:
    image_array = np.array(img_resized).astype("float32")
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch, img

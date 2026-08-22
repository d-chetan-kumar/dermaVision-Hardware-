from PIL import Image
import numpy as np

def is_valid_dermatology_image(image):
    """
    Validates if the input is a valid image suitable for inference.
    Does not reject valid ISIC images based on dark circular borders.
    """
    try:
        # Load image
        if hasattr(image, "path"):
            img = Image.open(image.path)
        elif isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.open(image)
            
        img = img.convert('RGB')
        
        # Check valid dimensions
        width, height = img.size
        if width < 50 or height < 50:
            return False
            
        img_arr = np.array(img)
        
        # Check completely black or white
        if img_arr.max() == 0 or img_arr.min() == 255:
            return False
            
        # Standard deviation check to ensure it's not a single solid color
        if np.std(img_arr) < 2.0:
            return False
            
        return True
    except Exception:
        return False

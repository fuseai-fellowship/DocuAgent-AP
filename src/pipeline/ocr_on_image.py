

import time
from src.models.paddle_models import ocr_model


def ocr_on_image(image, image_name):
    """
    Perform Optical Character Recognition on an image.
    
    Extracts text content from the input image using PaddleOCR. The function
    detects text regions, recognizes characters, and returns structured results
    with text content, bounding box coordinates, and confidence scores.
    
    Args:
        image (np.ndarray): Input image in OpenCV format (BGR) to perform OCR on.
        image_name (str): Identifier/name of the image being processed (for logging).
    
    Returns:
        list: OCR prediction results containing:
            - Detected text regions with bounding box coordinates
            - Recognized text content
            - Confidence scores for each detection
            Structure: [[[coordinates], (text, confidence)], ...]
    
 
    Note:
        - Works best on preprocessed/cleaned images
        - Performance depends on image quality and text complexity
        - Should be run on masked images to avoid OCR on non-text regions
    """
    # Initialize OCR model
    ocr = ocr_model()
    
    print(f"[INFO] Performing OCR on detected regions for {image_name}")
    
    # Track OCR processing time
    start_time = time.time()
    
    # Perform OCR prediction
    results = ocr.predict(image)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"[INFO] OCR on {image_name} took {elapsed:.2f} seconds")
    
    return results
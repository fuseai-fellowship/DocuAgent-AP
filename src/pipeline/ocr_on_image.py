
from pathlib import Path
import time
from src.models.paddle_models import ocr_model

def ocr_on_image(image, image_name):

    ocr = ocr_model()
    print(f"[INFO] Performing OCR on detected regions for {image_name}")
    start_time = time.time()
    results = ocr.predict(image)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] OCR on {image_name} took {elapsed:.2f} seconds")

    return results



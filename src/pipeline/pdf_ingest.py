from pathlib import Path
import cv2
import time, fitz # PyMuPDF

import numpy as np
from src.pipeline.image_pipeline import image_pipeline
from src.utils.paths import RAW


def pdf_to_images(pdf_path, dpi=150):
    pdf_name = Path(pdf_path).stem
   
    save_dir = RAW / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        print("[INFO] PDF opened successfully.")
    except Exception as e:
        raise ValueError(f"Error opening PDF: {e}")

    print(f"[INFO] Converting PDF to images for {pdf_name}")
    start_time = time.time()
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
             # convert to numpy array opencv format
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # if RGBA, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # if RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          
            yield i, img

    except Exception as e:
        print(f"[ERROR] Failed to convert PDF to images: {e}")
    finally:
        doc.close()
        end_time = time.time()
        print(f"[INFO] PDF to image conversion took {end_time - start_time:.2f} seconds")

def process_pdf_pipeline(pdf_path, preprocess=False, layout_result_save =False, dpi = 150):
   
    pdf_name = Path(pdf_path).stem
    start_time = time.time()
    try:
        for i, img in pdf_to_images(pdf_path, dpi=dpi):
            page_start_time = time.time()
            print(f"[INFO] Processing page {i+1} of {pdf_name}")
            
            image_pipeline(
                img=img,
                pdf_name=pdf_name,
                preprocess=preprocess,
                layout_result_save=layout_result_save,
                pdf_page_num=i+1
            )
            
            page_end_time = time.time()
            print(f"[INFO] Processing page {i+1} took {page_end_time - page_start_time:.2f} seconds")
    
    finally:
        end_time = time.time()
        print(f"[INFO] Pipeline on {pdf_name} took {end_time - start_time:.2f} seconds")
      

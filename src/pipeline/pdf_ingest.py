from pathlib import Path
import time
from src.pipeline.pdf_to_image import pdf_to_images
from src.utils.paths import OUTPUT

from src.pipeline.image_pipeline import image_pipeline


def process_pdf_pipeline(pdf_path, preprocess=False, layout_result_save =False):
   
    pdf_name = Path(pdf_path).stem
    
    # convert pdf to images and save them locally
    image_paths = pdf_to_images(pdf_path)
    
    start_time = time.time()    
    for i, img_path in enumerate(image_paths):
        
        page_start_time = time.time()
        print(f"[INFO] Processing page {i+1} of {pdf_name}")
        image_pipeline(img_path, pdf_name, preprocess, layout_result_save,pdf_page_num=i+1)

        page_end_time = time.time()
        page_elapsed = page_end_time - page_start_time
        print(f"[INFO] Processing page {i+1} took {page_elapsed:.2f} seconds")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] Pipeline on {pdf_name} took {elapsed:.2f} seconds")

        
      

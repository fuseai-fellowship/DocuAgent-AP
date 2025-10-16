from pathlib import Path
import time, fitz # PyMuPDF


from src.pipeline.image_pipeline import image_pipeline
from src.utils.paths import RAW


def pdf_to_images(pdf_path):
    pdf_name = Path(pdf_path).stem
   
    save_dir = RAW / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    print("[INFO] PDF opened successfully.")

    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  
        try:
            img_path = save_dir / f"page_{i+1}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"Error converting pdf to images in page {i+1}: {e}")
    doc.close()
    return image_paths


def process_pdf_pipeline(pdf_path, preprocess=False, layout_result_save =False, ocr_img_save = False):
   
    pdf_name = Path(pdf_path).stem
    
    # convert pdf to images and save them locally
    image_paths = pdf_to_images(pdf_path)
    
    start_time = time.time()    
    for i, img_path in enumerate(image_paths):
        
        page_start_time = time.time()
        print(f"[INFO] Processing page {i+1} of {pdf_name}")
        image_pipeline(img_path=img_path,
                       pdf_name=pdf_name,
                       preprocess=preprocess,
                       layout_result_save=layout_result_save,
                       pdf_page_num=i+1,
                       ocr_img_save=ocr_img_save
                       )

        page_end_time = time.time()
        page_elapsed = page_end_time - page_start_time
        print(f"[INFO] Processing page {i+1} took {page_elapsed:.2f} seconds")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] Pipeline on {pdf_name} took {elapsed:.2f} seconds")

        
      

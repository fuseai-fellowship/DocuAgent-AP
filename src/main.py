from src.pipeline.image_pipeline import image_pipeline
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.utils.paths import RAW

import cv2 
if __name__ == "__main__":
  
    sample_pdf4 = RAW/ "sample_pdf4.pdf"
    process_pdf_pipeline(pdf_path=sample_pdf4, preprocess=True, layout_result_save=True)
    print(f"OCR results for img is done")
    
    sample_img1 = RAW/ "sample_image1.png"
    image_pipeline(img_path= sample_img1, preprocess=True, layout_result_save=True)
    print(f"OCR results for img is done")
    
    sample_img2 = RAW/ "sample_image2.png"
    image_pipeline(img_path=sample_img2, preprocess=True, layout_result_save=True, )
    print(f"OCR results for img is done")

    sample_img4 = RAW / "sample_image4.png"
    image_pipeline(img_path=sample_img4, preprocess=True, layout_result_save=True)
    print(f"OCR results for img is done")

    sample_pdf2 = RAW / "sample_pdf2.pdf"
    process_pdf_pipeline(pdf_path= sample_pdf2,layout_result_save=True, preprocess=True)
    print(f"OCR results for PDF pages is done")
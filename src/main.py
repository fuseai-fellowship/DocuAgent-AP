from src.pipeline.image_pipeline import image_pipeline
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.utils.paths import  DATA, OUTPUT
from src.pipeline.chat_with_LLM import chat_with_gemini

import cv2 
if __name__ == "__main__":
  
    # sample_chart = DATA/"example.png"
    # image_pipeline(img_path=sample_chart, preprocess=False, layout_result_save=True)
    # sample_pdf4 = DATA/ "sample_pdf1.pdf"
    # process_pdf_pipeline(pdf_path=sample_pdf4, preprocess=False, layout_result_save=True)
    # print(f"OCR results for img is done")
    
    # sample_img1 = DATA/ "sample_image1.png"
    # image_pipeline(img_path= sample_img1, preprocess=False, layout_result_save=True)
    # print(f"OCR results for img is done")
    
    # sample_img2 = DATA/ "sample_image2.png"
    # image_pipeline(img_path=sample_img2, preprocess=False, layout_result_save=True)
    # print(f"OCR results for img is done")

    # sample_img4 = DATA / "sample_image4.png"
    # image_pipeline(img_path=sample_img4, preprocess=False, layout_result_save=True)
    # print(f"OCR results for img is done")
    
    sample_img5 = DATA / "table1.jpg"
    image_pipeline(img_path=sample_img5, preprocess=False, layout_result_save=True)
    print(f"✅ OCR results for img is done")
    chat_with_gemini(OUTPUT / "table1" / "table1_all_results.json")

    # sample_pdf2 = DATA / "sample_pdf2.pdf"
    # process_pdf_pipeline(pdf_path= sample_pdf2,layout_result_save=True, preprocess=False)
    # print(f"OCR results for PDF pages is done")
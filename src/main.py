from src.pipeline.image_pipeline import image_pipeline
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.utils.paths import RAW
if __name__ == "__main__":
  
    sample_pdf2 = RAW / "sample_pdf2.pdf"
    pdf_results2 =process_pdf_pipeline(sample_pdf2,layout_result_save=True, preprocess=False) 
    print(f"OCR results for PDF pages is done")
    
    sample_img1 = RAW/ "sample_image1.png"
    img_results1 = image_pipeline(sample_img1, preprocess=False, layout_result_save=True)
    print(f"OCR results for img is done")
    
    sample_img2 = RAW/ "sample_image2.png"
    img_results2 = image_pipeline(sample_img2, preprocess=False, layout_result_save=True)
    print(f"OCR results for img is done")
    
 
    sample_img4 = RAW/ "sample_image4.png"
    img_results4 = image_pipeline(sample_img4, preprocess=False, layout_result_save=True)
    print(f"OCR results for img is done")

    # sample_pdf4 = RAW/ "sample_pdf4.pdf"
    # img_results5 = process_pdf_pipeline(sample_pdf4, preprocess=False, layout_result_save=True)
    # print(f"OCR results for img is done")
    
    
    
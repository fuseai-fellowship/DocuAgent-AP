
from pathlib import Path
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import paddle, numpy as np, json, time


# Local imports
from src.utils.paths import RAW, PROCESSED, OUTPUT


def run_ocr_on_image(image_path, ocr, pdf_name=None):
   
    try:
       paddle.set_device('gpu')  
    except Exception as e:
         print(f"[WARNING] Could not set device to GPU. Using default device. Error: {e}")
         
    image_name = Path(image_path).stem
    start_time = time.time()
    results = ocr.predict(str(image_path))
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] OCR on {image_name} took {elapsed:.2f} seconds")
    
    if pdf_name:
        ocr_img_dir = PROCESSED/ pdf_name
    else:
         ocr_img_dir = PROCESSED / image_name
    ocr_img_dir.mkdir(parents=True, exist_ok=True)
    structured = {}
    for res in results:

        # to store processed image with bounding boxes
        res.save_to_img(str(ocr_img_dir))
        structured["recognized_texts"] = res['rec_texts']
        structured["recognized_confidence"] = [float(score) for score in res['rec_scores']]
        
        # bounding boxes are in np.ndarray format; converted to list for JSON serialization
        structured["bounding_boxes"] = [
        poly.tolist() if isinstance(poly, np.ndarray) else poly
        for poly in res['rec_polys']
    ] 
    
     # aves JSON inside outputs/image_name/ if not part of a PDF
    if pdf_name==None:
         save_dir = OUTPUT / image_name
         save_dir.mkdir(parents=True, exist_ok=True)

         output_json = save_dir / f"{image_name}_result.json"
         with open(output_json, "w", encoding="utf-8") as f:
             json.dump(structured, f, indent=2, ensure_ascii=False)
    return structured
    


def pdf_to_images(pdf_path, pdf_name):
  
    save_dir = RAW / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  
        img_path = save_dir / f"page_{i+1}.png"
        pix.save(str(img_path))
        image_paths.append(str(img_path))

    return image_paths


def process_pdf(pdf_path, ocr):
   
    pdf_name = Path(pdf_path).stem

    # convert pdf to images and save them locally
    image_paths = pdf_to_images(pdf_path, pdf_name)

    # ocr on each image and collect results
    all_pages_results = {}
    for i, img_path in enumerate(image_paths):
        page_results = run_ocr_on_image(img_path, ocr, pdf_name=pdf_name)
        all_pages_results[f"page_{i+1}"] = page_results

    #  Save JSON inside outputs/pdf_name/
    save_dir = OUTPUT / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    output_json = save_dir / f"{pdf_name}_result.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_pages_results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] OCR results saved to {output_json}")
    return all_pages_results


if __name__ == "__main__":
    # to initialize OCR once
    
    #---- text image preprocessing + text detection + using mobile model ------
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    #---- text detection + text recognition using server model ----

    # ocr = PaddleOCR(
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False)

    #---- text image preprocessing + text detection + textline orientation classification + text recognition ------

    # ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True)
   
   
    sample_image1 = RAW / "sample_image1.png"
    image_results1 = run_ocr_on_image(sample_image1, ocr)
    print(f"OCR results for image: {image_results1}")  

    sample_image2 = RAW / "sample_image2.png"
    image_results2 = run_ocr_on_image(sample_image2, ocr)
    print(f"OCR results for image: {image_results2}")  

    sample_image3 = RAW / "sample_image3.jpg"
    image_results3 = run_ocr_on_image(sample_image3, ocr)
    print(f"OCR results for image: {image_results3}")

    sample_pdf1 = RAW / "sample_pdf1.pdf"
    pdf_results1 = process_pdf(sample_pdf1, ocr)
    print(f"OCR results for PDF pages: {list(pdf_results1.keys())}")
    
    sample_pdf2 = RAW / "sample_pdf2.pdf"
    pdf_results2 = process_pdf(sample_pdf2, ocr)    
    print(f"OCR results for PDF pages: {list(pdf_results2.keys())}")
    
    sample_pdf3 = RAW / "sample_pdf3.pdf"
    pdf_results3 = process_pdf(sample_pdf3, ocr)    
    print(f"OCR results for PDF pages: {list(pdf_results3.keys())}")
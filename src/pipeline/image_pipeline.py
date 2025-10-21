import time, cv2, random, numpy as np
from pathlib import Path

from src.utils.paths import OUTPUT
from src.pipeline.layout_on_image import layout_on_image
from src.pipeline.image_preprocessing import preprocess_document
from src.pipeline.ocr_on_image import ocr_on_image
from src.pipeline.masking_on_image import mask_regions
from src.utils.save_results import save_results

def image_pipeline(
    img=None,
    img_path=None,
    pdf_name=None,
    preprocess=True,
    layout_result_save=False,
   
    pdf_page_num=1,

):
    if img is None and img_path is None:
        raise ValueError("Either img or img_path must be provided.")

    # --- load image if path is provided ---
    if img is None:
        img = cv2.imread(str(img_path))
        print("[INFO] Image loaded:", img_path)

    # --- Output naming ---
    if img_path:
        img_name = Path(img_path).stem
    elif pdf_name:
        img_name = f"{pdf_name}_page_{pdf_page_num}"
    else:
        img_name = str(random.randint(1000, 9999))
 
    file_name = pdf_name if pdf_name else img_name  # for output dirs

    print(f"\n[INFO] Starting image pipeline for: {img_name}")
    image_start = time.time()

    # --- Step 1: Preprocess ---
    if preprocess:
        
        preprocess_document(img)


    # --- Step 2: Layout Detection ---
    layout_results = layout_on_image(img, img_name)
 
    # --- Step 3: Masking ---
    masked_img = mask_regions(img, img_name, layout_results)
    # --- Step 4: OCR ---
    ocr_results = ocr_on_image(masked_img, img_name)
    
    structured_ocr_results = {}
    for res in ocr_results:
        structured_ocr_results["rec_polys"] = [
            poly.tolist() if isinstance(poly, np.ndarray) else poly
            for poly in res['rec_polys']
            ]
        structured_ocr_results["rec_scores"] = [float(score) for score in res['rec_scores']]
        structured_ocr_results["rec_texts"] = res['rec_texts']

    if layout_result_save:
        save_results(OUTPUT=OUTPUT,
                     file_name=file_name,
                     img_name=img_name,
                     img=img,
                     masked_img=masked_img,
                     page_num=pdf_page_num,
                     layout_results=layout_results,
                     layout_result_save=layout_result_save,
                     original_ocr_results=ocr_results,
                     ocr_results=structured_ocr_results)
    else:
        save_results(OUTPUT=OUTPUT,
                     file_name=file_name,
                     img_name=img_name,
                     img=img,
                     masked_img=masked_img,
                     page_num=pdf_page_num,
                     layout_results=None,
                     original_ocr_results=ocr_results,
                     layout_result_save=layout_result_save,
                     ocr_results=structured_ocr_results)
    image_end = time.time()
    print(f"[TOTAL TIME] {img_name}: {image_end - image_start:.2f}s\n")



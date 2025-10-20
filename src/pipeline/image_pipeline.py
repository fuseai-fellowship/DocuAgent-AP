import time, cv2, json, numpy as np
from pathlib import Path
from src.models.paddle_models import ocr_model
from src.pipeline.layout_on_image import layout_on_image
from src.pipeline.image_preprocessing import preprocess_document
from src.utils.paths import OUTPUT
from src.models.paddle_models import formula_model
import random

def preprocess_region(img, x1, y1, x2, y2, pad_w=5, pad_h=5, min_height=32):
      # Add padding with boundary checks
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(img.shape[1], x2 + pad_w)
    y2_pad = min(img.shape[0], y2 + pad_h)
    
    # Crop region
    region = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Upscale if region is too small
    h, w = region.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_width = int(w * scale)
        region = cv2.resize(region, (new_width, min_height), interpolation=cv2.INTER_LINEAR)
   
    return region


def image_pipeline(img=None, img_path=None, pdf_name=None, preprocess=True, layout_result_save=False, pdf_page_num=1, ocr_img_save=False):
        
    if img is None and img_path is None:
        raise ValueError("Either img or img_path must be provided.")
    
    if img is None and img_path is not None:
        img = cv2.imread(str(img_path))
        print("[INFO] Image loaded from path:", img_path)
    

    if img_path:
        img_name = Path(img_path).stem
    elif pdf_name:
        img_name = f"{pdf_name}_page_{pdf_page_num}"
    else:
        img_name = str(random.randint(1000, 9999))
        
    file_name = pdf_name if pdf_name else img_name  # To create save directories
  
    SKIP_FOR_OCR = ["table", "figure_table_title", "chart", "image", "seal", "formula"]
    img_start_time = time.time()

    # Change: logging updated for in-memory images
    print(f"[INFO] Starting processing for image: {img_name}")

    # Change: only read from disk if img_path is provided and preprocess is True
    if preprocess:
      
        print(f"[INFO] Preprocessing enabled for {img_name}")
        before_preprocess_time = time.time()
        # img = preprocess_document(img)

        after_preprocess_time = time.time()
        print(f"[INFO] Image Preprocessing took {after_preprocess_time - before_preprocess_time:.2f} seconds")

    # Change: layout detection now works on in-memory image
    
    layout_results = layout_on_image(img, image_name=img_name)

    if layout_result_save:
        layout_dir = OUTPUT / file_name / "layout"
        layout_dir.mkdir(parents=True, exist_ok=True)

        for res in layout_results:
            layout_img = layout_dir / f"{img_name}_layout.png"
            layout_json = layout_dir / f"{img_name}_layout.json"
            res.save_to_json(str(layout_json))
            res.save_to_img(str(layout_img))

    ocr_json_dir = OUTPUT / file_name / "ocr/json"
    formula_dir = OUTPUT / file_name / "formula"
    ocr_json_dir.mkdir(parents=True, exist_ok=True)
    formula_dir.mkdir(parents=True, exist_ok=True)

    if ocr_img_save:
        ocr_img_dir = OUTPUT / file_name / "ocr/images"
        ocr_img_dir.mkdir(parents=True, exist_ok=True)

    all_ocr_results = {}

    # Change: img already loaded in-memory, no need to read again
    ocr = ocr_model()
    before_ocr_time = time.time()
    print(f"[INFO] Performing OCR on detected regions for {img_name}")
    
    
    for page_result in layout_results:
          for i, box in enumerate(page_result["boxes"]):
            cls = box['label']
            conf = box['score']
            coords = [int(x) for x in box['coordinate']]
            x1, y1, x2, y2 = coords
            
            region = preprocess_region(img, x1, y1, x2, y2)
            # --- OCR Regions preprocessed and cropped 
            if (cls not in SKIP_FOR_OCR):
                ocr_result = ocr.predict(region)
                
                if ocr_img_save:
                    for res in ocr_result:
                        res.save_to_img(str(ocr_img_dir))
                # Store OCR results per region
            
                if f"region_{i}" not in all_ocr_results:
                      all_ocr_results[f"region_{i}"] = []
                
                for res in ocr_result:
                    structured = {}
                    structured["class"] = cls
                    structured["confidence"] = float(conf)
                    structured["bbox"] = [x1, y1, x2, y2]  # original region box
                    structured["recognized_texts"] = res['rec_texts']
                    structured["recognized_confidence"] = [float(score) for score in res['rec_scores']]
                    
                    ##### ----------------- COORDINATE POINTS OF CROPPED IMAGE ARE ADDED TO BBOX COORDINATES  ------------- 
                    for poly in res['rec_polys']:
                        for p in poly:
                            p[0] = float(p[0] + x1)
                            p[1] = float(p[1] + y1)

                            pt1 = (int(p[0]), int(p[1]))
                            pt2 = (int(p[0]+1), int(p[1]+1))  # small rectangle for visibility
                            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)  # green 1px

                    for poly in res['rec_polys']:
                        pts = np.array(poly, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # red polygon

                    # json serializable format
                    structured["recognized_polys"] = [
                        poly.tolist() if isinstance(poly, np.ndarray) else poly
                        for poly in res['rec_polys']
                    ]

                    all_ocr_results[f"region_{i}"] = structured

            # --- formula Regions evaluated by another model ---
            elif cls == "formula":
                formula_output = formula_model.predict(region)
                for idx, f_res in enumerate(formula_output):
                    formula_img_path = formula_dir / f"{img_name}_formula_{i}_{idx}.png"
                    formula_json_path = formula_dir / f"{img_name}_formula_{i}_{idx}.json"
                    f_res.save_to_img(save_path=str(formula_img_path))
                    f_res.save_to_json(save_path=str(formula_json_path))
        
    after_ocr_time = time.time()
    print(f"[INFO] OCR on regions for {img_name} took {after_ocr_time - before_ocr_time:.2f} seconds")

    ocr_json_path = ocr_json_dir / f"{file_name}_ocr.json"
    if ocr_img_save:
         annotated_img_path = ocr_img_dir / f"{img_name}_ocr_polygons.png"
         cv2.imwrite(str(annotated_img_path), img)
    # to make sure file is empty
    if not ocr_json_path.exists():
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    #read the json content 
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)

    # updated with each region results
    current_data[f"page_{pdf_page_num}"] = all_ocr_results
 
    # write to file
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

    img_end_time = time.time()
    print(f"[INFO] Total processing time for image {img_name} took {img_end_time - img_start_time:.2f} seconds")
    print(f"[INFO] OCR results saved for page {pdf_page_num}")


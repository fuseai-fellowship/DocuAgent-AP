
import time, cv2, json, numpy as np
from pathlib import Path
from src.models.paddle_models import ocr_model
from src.pipeline.layout_on_image import layout_on_image
from src.pipeline.ocr_on_image import ocr_on_image


from src.utils.paths import OUTPUT
from src.models.paddle_models import formula_model

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

def normalize_full_page_output(ocr_result):
    all_ocr_result = {}
    
    rec_texts = ocr_result["rec_texts"]
    rec_scores = ocr_result["rec_scores"]
    rec_polys = ocr_result["rec_polys"]

     # convert each polygon (4 points) to bounding box [x1, y1, x2, y2]
    bounding_boxes = []
    for poly in rec_polys:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        bounding_boxes.append([min(xs), min(ys), max(xs), max(ys)])

    # Enumerate and build structured results per region
    for i, (rec_score, rec_text, poly) in enumerate(zip(rec_scores, rec_texts, rec_polys)):
        structured = {}
        structured["class"] = "full_page"
        structured["confidence"] = float(rec_score)
        structured["bbox"] = [int(v) for v in bounding_boxes[i]]   # normalized per region
        structured["recognized_texts"] = [rec_text]
        structured["recognized_confidence"] = [float(rec_score)]
        structured["recognized_polys"] = [
            pt.tolist() if isinstance(pt, np.ndarray) else pt for pt in poly
        ]  # ensure polys are lists

        all_ocr_result[f"region_{i}"] = structured

    return all_ocr_result


def image_pipeline(img_path, pdf_name=False, preprocess=False, layout_result_save=False,pdf_page_num=1):


    image_name = Path(img_path).stem
    
    SKIP_FOR_OCR = ["table", "figure_table_title", "chart", "image", "seal", "formula"]  
    
    # Step 1: Layout Detection
    if preprocess:
        pass
    else:
        layout_results = layout_on_image(img_path)
   
    if layout_result_save:     
                
        layout_dir = OUTPUT / pdf_name / "layout" if pdf_name else OUTPUT / image_name / "layout"
        layout_dir.mkdir(parents=True, exist_ok=True)
    
        for res in layout_results:
            layout_img = layout_dir / f"{image_name}_layout.png"
            layout_json = layout_dir / f"{image_name}_layout.json"
            res.save_to_img(str(layout_img))
            res.save_to_json(str(layout_json))
        
    # Step 2: OCR on useful regions
    detected_classes = []
    for page_result in layout_results:
        for box in page_result["boxes"]:
            detected_classes.append(box["label"])

    file_name = pdf_name if pdf_name else image_name
    ocr_img_dir = OUTPUT / file_name / "ocr/img"
    ocr_json_dir = OUTPUT / file_name / "ocr/json"
    formula_dir = OUTPUT / file_name / "formula"
    
    ocr_img_dir.mkdir(parents=True, exist_ok=True)
    ocr_json_dir.mkdir(parents=True, exist_ok=True)
    formula_dir.mkdir(parents=True, exist_ok=True)
    
    all_ocr_results = {}    
    
    # Case 1: If no charts or table then do full page OCR
    if not any(cls in SKIP_FOR_OCR for cls in detected_classes):
        print("No charts or tables detected, performing full page OCR.")
        
        ocr_result = ocr_on_image(img_path)
        for res in ocr_result:
            res.save_to_img(str(ocr_img_dir))
            all_ocr_results= normalize_full_page_output(res)  
        
    else:
        img = cv2.imread(str(img_path))
        ocr = ocr_model()
        before_ocr_time = time.time()
        print(f"[INFO] Performing OCR on detected regions for {image_name}")
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
                out_img = ocr_img_dir / f"{image_name}_{cls}_{i}.png"
                # cv2.imwrite(str(out_img), region)
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
                    
                    ##### ----------------- COORDINATE POINTS ARE RELATIVE TO THE BBOX ------------- 
                    structured["recognized_polys"] = [
                              poly.tolist() if isinstance(poly, np.ndarray) else poly
                                for poly in res['rec_polys']
                      ] 
                    ##### --------------------------------------------------------------------
                    all_ocr_results[f"region_{i}"] = structured

            # --- formula Regions evaluated by another model ---
            elif cls == "formula":
                formula_output = formula_model.predict(region)
                for idx, f_res in enumerate(formula_output):
                    formula_img_path = formula_dir / f"{image_name}_formula_{i}_{idx}.png"
                    formula_json_path = formula_dir / f"{image_name}_formula_{i}_{idx}.json"
                    f_res.save_to_img(save_path=str(formula_img_path))
                    f_res.save_to_json(save_path=str(formula_json_path))
        
        after_ocr_time = time.time()
        print(f"[INFO] OCR on regions for {image_name} took {after_ocr_time - before_ocr_time:.2f} seconds")

    ocr_json_path = ocr_json_dir / f"{file_name}_ocr.json"

    # to make sure file is empty
    if not ocr_json_path.exists():
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    #read the json content 
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)

    # update with new page data
    current_data[f"page_{pdf_page_num}"] = all_ocr_results

    # write back to file
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

    print(f"[INFO] OCR results saved for page {pdf_page_num}")



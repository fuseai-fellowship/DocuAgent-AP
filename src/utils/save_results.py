import cv2, json

def save_results(OUTPUT,file_name,img_name, img, page_num, masked_img,ocr_results,original_ocr_results,layout_results=None,layout_result_save=False):
    
    output_path = OUTPUT / file_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessed_img_path = output_path / f"{img_name}_preprocessed.png"
    masked_img_path = output_path / f"{img_name}_masked.png"
    
    cv2.imwrite(str(preprocessed_img_path), img)
    cv2.imwrite(str(masked_img_path), masked_img)
    if layout_result_save == True and layout_results is not None:
    # Save layout results
        layout_dir = output_path / "layout"
        layout_dir.mkdir(parents=True, exist_ok=True)

        layout_img_path = layout_dir / f"{img_name}_layout.png"
        layout_json_path = layout_dir / f"{img_name}_layout.json"
    
        for res in layout_results:
            res.save_to_json(str(layout_json_path))
            try:
                res.save_to_img(str(layout_img_path))
            except TypeError as e:
                print(f"[WARNING] Skipping layout visualization for {file_name} due to color format issue: {e}")


    # Save OCR results
    ocr_dir = output_path / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    
    ocr_json_path = ocr_dir / f"{file_name}_ocr.json"
    ocr_img_path = ocr_dir / f"{img_name}_ocr.png"

    # Load existing JSON if it exists
    for res in original_ocr_results:
        res.save_to_img(str(ocr_img_path))
        res.save_to_json(ocr_dir / f"{img_name}_ocr_pagewise.json")
        
    if ocr_json_path.exists():
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            try:
                all_ocr_results = json.load(f)
            except json.JSONDecodeError:
                all_ocr_results = {}
    else:
        all_ocr_results = {}

    # Add or update the page
    all_ocr_results[f"page_{page_num}"] = ocr_results

    # Write back to file
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(all_ocr_results, f, indent=4, ensure_ascii=False)

    print(f"[INFO] OCR results saved for page {page_num} → {ocr_json_path}")

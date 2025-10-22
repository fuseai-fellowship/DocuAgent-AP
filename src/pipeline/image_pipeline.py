import time, cv2, random, numpy as np
from pathlib import Path
from src.pipeline.chart_extraction import run_chart2table
from src.pipeline.table_extraction import run_table_recognition
from src.utils.paths import OUTPUT
from src.pipeline.layout_on_image import layout_on_image
from src.pipeline.image_preprocessing import preprocess_document
from src.pipeline.ocr_on_image import ocr_on_image
from src.pipeline.masking_on_image import mask_regions
from src.utils.save_results import save_results, save_layout_json, save_table_html, save_chart_json
import json

def preprocess_region(img, x1, y1, x2, y2, pad_w=20, pad_h=20, min_height=32):
    # Add padding with boundary checks
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(img.shape[1], x2 + pad_w)
    y2_pad = min(img.shape[0], y2 + pad_h)

    return img[y1_pad:y2_pad, x1_pad:x2_pad]


def image_pipeline(img=None, img_path=None, pdf_name=None, preprocess=True, layout_result_save=False, pdf_page_num=1, ocr_img_save=False):
    if img is None and img_path is None:
        raise ValueError("Either img or img_path must be provided.")

    # --- Load image ---
    if img is None:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        print("[INFO] Image loaded:", img_path)

    # --- Naming ---
    img_name = Path(img_path).stem if img_path else f"{pdf_name}_page_{pdf_page_num}" if pdf_name else str(random.randint(1000, 9999))
    file_name = pdf_name if pdf_name else img_name

    print(f"\n[INFO] Starting image pipeline for: {img_name}")
    start_time = time.time()

    # --- Preprocess ---
    if preprocess:
        img = preprocess_document(img)  # make sure it returns the processed image

    # --- Layout Detection ---
    layout_results = layout_on_image(img, img_name)

    # --- Save Layout ---
    if layout_result_save:
        save_layout_json(file_name, img_name, layout_results)
        
    # --- Masking ---
    masked_img = mask_regions(img, img_name, layout_results)

    # Convert masked image to 3 channels if needed (for PaddleOCR)
    if len(masked_img.shape) == 2:
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
    elif masked_img.shape[2] == 1:
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)

    # --- OCR ---
    ocr_results = ocr_on_image(masked_img, img_name)
    structured_ocr_results = {}
    for res in ocr_results:
        structured_ocr_results["rec_polys"] = [
            poly.tolist() if isinstance(poly, np.ndarray) else poly
            for poly in res['rec_polys']
        ]
        structured_ocr_results["rec_scores"] = [float(score) for score in res['rec_scores']]
        structured_ocr_results["rec_texts"] = res['rec_texts']

    # --- Save Cropped Regions ---
    crop_save_dir = OUTPUT / file_name / "crops"
    crop_save_dir.mkdir(parents=True, exist_ok=True)

    for page_result in layout_results:
        for i, box in enumerate(page_result["boxes"]):
            x1, y1, x2, y2 = [int(c) for c in box['coordinate']]
            cls = box['label']
            region = preprocess_region(img, x1, y1, x2, y2)

            # --- Text blocks ---
            if cls in ["text", "paragraph_title"]:
                texts_in_box = []
                for idx, poly in enumerate(structured_ocr_results["rec_polys"]):
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    cx, cy = np.mean(xs), np.mean(ys)
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        texts_in_box.append(structured_ocr_results["rec_texts"][idx])
                box["block_content"] = " ".join(texts_in_box) if texts_in_box else ""

            # --- Image / Chart / Table ---
            elif cls in ["image", "chart", "table"]:
                crop_path = crop_save_dir / f"{img_name}_region_{i}_{cls}.png"
                cv2.imwrite(str(crop_path), region)
                print(f"[INFO] Saved cropped {cls} region: {crop_path}")

                if cls == "image":
                    box["block_content"] = str(crop_path)

                elif cls == "table":
                    cv2.imwrite(str(crop_path), region)
                    print(f"[INFO] Saved cropped {cls} region: {crop_path}")

                    table_output_dir = crop_save_dir / "table_html"
                    table_output_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        run_table_recognition(
                            image_path=str(crop_path),
                            output_path=str(table_output_dir)
                        )

                        save_table_html(table_output_dir, box)
                    except Exception as e:
                        print(f"[ERROR] Table recognition failed for {crop_path}: {e}")
                        box["block_content"] = None

                elif cls == "chart":
                    chart_output_dir = crop_save_dir / "chart_json"
                    chart_output_dir.mkdir(parents=True, exist_ok=True)
                    chart_json_path = chart_output_dir / f"{img_name}_region_{i}_{cls}.json"
                    try:
                        run_chart2table(
                            image_path=str(crop_path),
                            output_path=str(chart_json_path)
                        )
                        print(f"[INFO] Chart2Table JSON saved: {chart_json_path}")
                        save_chart_json(chart_json_path, box)
                    except Exception as e:
                        print(f"[ERROR] Chart2Table failed for {crop_path}: {e}")
                        box["block_content"] = None

    # --- Save results ---
    save_results(
        OUTPUT=OUTPUT,
        file_name=file_name,
        img_name=img_name,
        img=img,
        masked_img=masked_img,
        page_num=pdf_page_num,
        layout_results=layout_results if layout_result_save else None,
        original_ocr_results=ocr_results,
        layout_result_save=layout_result_save,
        ocr_results=structured_ocr_results
    )

    end_time = time.time()
    print(f"[TOTAL TIME] {img_name}: {end_time - start_time:.2f}s\n")
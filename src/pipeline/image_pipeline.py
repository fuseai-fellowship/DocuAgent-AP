"""
Document Image Processing Pipeline
Handles preprocessing, layout detection, OCR, table extraction, and chart extraction.
"""

import time
import cv2
import random
import numpy as np
from pathlib import Path

from src.pipeline.chart_extraction import run_chart2table
from src.pipeline.table_extraction import run_table_recognition
from src.pipeline.layout_on_image import layout_on_image
from src.pipeline.image_preprocessing import preprocess_document
from src.pipeline.ocr_on_image import ocr_on_image
from src.pipeline.masking_on_image import mask_regions
from src.utils.paths import OUTPUT
from src.utils.save_results import (
    save_results,
    save_cropped_img,
    save_table_html,
    save_chart_json,
    create_output_dir
)


# Configuration constants
SKIP_FOR_OCR = ["table", "chart", "image", "seal", "formula"]
MIN_REGION_HEIGHT = 32
PADDING_WIDTH = 5
PADDING_HEIGHT = 12


def preprocess_region(img, x1, y1, x2, y2, pad_w=PADDING_WIDTH, pad_h=PADDING_HEIGHT, min_height=MIN_REGION_HEIGHT):
    """
    Preprocess a specific region of the image with padding and upscaling.
    
    Args:
        img: Input image array
        x1, y1, x2, y2: Bounding box coordinates
        pad_w: Horizontal padding (default: 5)
        pad_h: Vertical padding (default: 5)
        min_height: Minimum height threshold for upscaling (default: 32)
    
    Returns:
        Preprocessed region as numpy array
    """
    # Add padding with boundary checks
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(img.shape[1], x2 + pad_w)
    y2_pad = min(img.shape[0], y2 + pad_h)
    
    # Crop region
    region = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Upscale if region is too small for better processing
    h, w = region.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_width = int(w * scale)
        region = cv2.resize(region, (new_width, min_height), interpolation=cv2.INTER_LINEAR)
   
    return region


def structure_ocr_results(ocr_results):
    """
    Convert raw OCR results to structured format with serializable data types.
    
    Args:
        ocr_results: Raw OCR results from OCR engine
    
    Returns:
        Dictionary with structured OCR data (polygons, scores, texts)
    """
    structured_results = {}
    
    for res in ocr_results:
        # Convert numpy arrays to lists for JSON serialization
        structured_results["rec_polys"] = [
            poly.tolist() if isinstance(poly, np.ndarray) else poly
            for poly in res['rec_polys']
        ]
        structured_results["rec_scores"] = [float(score) for score in res['rec_scores']]
        structured_results["rec_texts"] = res['rec_texts']
    
    return structured_results


def extract_text_from_box(box_coords, structured_ocr_results):
    """
    Extract text content that falls within a bounding box.
    
    Args:
        box_coords: Tuple of (x1, y1, x2, y2) coordinates
        structured_ocr_results: Structured OCR results dictionary
    
    Returns:
        String of concatenated text within the box
    """
    x1, y1, x2, y2 = box_coords
    texts_in_box = []
    
    for idx, poly in enumerate(structured_ocr_results["rec_polys"]):
        # Calculate centroid of text polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = np.mean(xs), np.mean(ys)
        
        # Check if centroid is within box bounds
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            texts_in_box.append(structured_ocr_results["rec_texts"][idx])
    
    return " ".join(texts_in_box) if texts_in_box else ""


def process_table_region(region, img_name, region_num, table_dir):
    """
    Process a table region and extract structured HTML content.
    
    Args:
        region: Cropped image region containing table
        img_name: Name of the source image
        region_num: Region index number
        table_dir: Table directory path (from dirs dict)
    
    Returns:
        Extracted table HTML string or None if extraction fails
    """
    try:
        table_results = run_table_recognition(region, img_name)
        save_table_html(table_results, img_name, region_num, table_dir)
        
        # Extract HTML content from results
        extracted_content = [res.html for res in table_results]
        if extracted_content and "table_1" in extracted_content[0]:
            return str(extracted_content[0]["table_1"])
        else:
            print(f"[WARNING] No table content found in region {region_num}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Table recognition failed for region {region_num}: {e}")
        return None


def process_chart_region(region, img_name, region_num, chart_dir):
    """
    Process a chart region and extract structured data.
    
    Args:
        region: Cropped image region containing chart
        img_name: Name of the source image
        region_num: Region index number
        chart_dir: Chart directory path (from dirs dict)
    
    Returns:
        Extracted chart data as string or None if extraction fails
    """
    try:
        chart_results = run_chart2table(image=region, image_name=img_name)
  
         # Extract chart data from results
        extracted_content = None
        for res in chart_results:
            extracted_content = f"{res["result"]}"
        # Convert generator to list if needed
        save_chart_json(extracted_content, img_name, region_num, chart_dir)
        if extracted_content:
            return extracted_content
        else:
            print(f"[WARNING] No chart data found in region {region_num}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Chart extraction failed for region {region_num}: {e}")
        return None


def ensure_3channel_image(img):
    """
    Convert image to 3-channel BGR format if needed.
    Required for some OCR engines like PaddleOCR.
    
    Args:
        img: Input image array
    
    Returns:
        3-channel BGR image
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def image_pipeline(
    img=None,
    img_path=None,
    pdf_name=None,
    preprocess=True,
    layout_result_save=False,
    pdf_page_num=1,
    dirs=None
):
    """
    Main image processing pipeline that orchestrates all document analysis steps.
    
    Args:
        img: Pre-loaded image array (optional)
        img_path: Path to image file (optional)
        pdf_name: Name of source PDF if processing PDF pages
        preprocess: Whether to apply preprocessing (default: True)
        layout_result_save: Whether to save layout visualization (default: False)
        pdf_page_num: Page number if processing PDF (default: 1)
        output_dir: Custom output directory (optional)
    
    Returns:
        None (saves all results to disk)
    
    Raises:
        ValueError: If neither img nor img_path is provided
        FileNotFoundError: If img_path cannot be read
    """
    
    # ========== Input Validation ==========
    if img is None and img_path is None:
        raise ValueError("Either img or img_path must be provided.")
    
    # ========== Load Image ==========
    if img is None:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        print(f"[INFO] Image loaded: {img_path}")
    
    # ========== Setup Naming and Directories ==========
    # Generate appropriate names based on input type
    if img_path:
        img_name = Path(img_path).stem
    elif pdf_name:
        img_name = f"{pdf_name}_page_{pdf_page_num}"
    else:
        img_name = f"image_{random.randint(1000, 9999)}"
    
    file_name = pdf_name if pdf_name else img_name
    
    # Create all output directories once and get directory paths
    if dirs is None:
        dirs = create_output_dir(file_name)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Starting pipeline for: {img_name}")
    print(f"[INFO] Output directory: {dirs['base']}")
    print(f"{'='*60}")
    start_time = time.time()
    
    # ========== Step 1: Preprocessing ==========
    if preprocess:
        print("[STEP 1/6] Preprocessing image...")
        img = preprocess_document(img)
    
    # ========== Step 2: Layout Detection ==========
    print("[STEP 2/6] Detecting layout...")
    layout_results = layout_on_image(img, img_name)
    
    # ========== Step 3: Masking Regions ==========
    print("[STEP 3/6] Masking non-text regions...")
    masked_img = mask_regions(img, img_name, layout_results, SKIP_FOR_OCR)
    masked_img = ensure_3channel_image(masked_img)
    
    # ========== Step 4: OCR ==========
    print("[STEP 4/6] Running OCR...")
    ocr_results = ocr_on_image(masked_img, img_name)
    structured_ocr_results = structure_ocr_results(ocr_results)
    
    # ========== Step 5: Process Regions ==========
    print("[STEP 5/6] Processing detected regions...")
    
    table_count = 0
    chart_count = 0
    
    for page_result in layout_results:
        for i, box in enumerate(page_result["boxes"]):
            # Extract box coordinates and label
            # Convert coordinates to integers and store back in box
            box['coordinate'] = [int(c) for c in box['coordinate']]
            x1, y1, x2, y2 = box['coordinate']
            cls = box['label']
            
            # Crop and preprocess region
            region = preprocess_region(img, x1, y1, x2, y2, pad_h=0, pad_w=0)
            region_num = i + 1
            
            # ===== Process Text Blocks =====
            if cls not in SKIP_FOR_OCR:
                box["block_content"] = extract_text_from_box(
                    (x1, y1, x2, y2),
                    structured_ocr_results
                )
            
            # ===== Process Special Regions (Image/Chart/Table) =====
            elif cls in ["image", "chart", "table"]:
                # Save cropped region
                crop_path = save_cropped_img(region, dirs['crop'], img_name, region_num)
                
                if cls == "image":
                    box["block_content"] = str(crop_path)
                
                elif cls == "table":
                    table_count += 1
                    box["block_content"] = process_table_region(
                        region, img_name, region_num, dirs['table']
                    )
                
                elif cls == "chart":
                    chart_count += 1
                    box["block_content"] = process_chart_region(
                        region, img_name, region_num, dirs['chart']
                    )
    
    # ========== Warnings for Empty Results ==========
    if table_count == 0:
        print("\n [WARNING] No tables detected or extracted in this image")
    else:
        print(f" [INFO] Successfully processed {table_count} table(s)")
    
    if chart_count == 0:
        print(" [WARNING] No charts detected or extracted in this image")
    else:
        print(f"[INFO] Successfully processed {chart_count} chart(s)")
    
    # ========== Step 6: Save Results ==========
    print("[STEP 6/6] Saving all results...")
    save_results(
        dirs=dirs,
        file_name=file_name,
        img_name=img_name,
        img=img,
        masked_img=masked_img,
        page_num=pdf_page_num,
        layout_results=layout_results,
        original_ocr_results=ocr_results,
        structured_ocr_results=structured_ocr_results,
        layout_result_save=layout_result_save
    )
    
    # ========== Pipeline Complete ==========
    end_time = time.time()
    print(f"\n{'='*60}")
    print(f" [SUCCESS] Pipeline completed for: {img_name}")
    print(f"  [TOTAL TIME FOR IMAGE] {end_time - start_time:.2f}s")
    print(f"{'='*60}\n")
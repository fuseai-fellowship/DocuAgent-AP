"""
Results Saving Utilities
Handles saving of all pipeline outputs with organized directory structure.
"""

import cv2
import json
from pathlib import Path
from src.utils.paths import OUTPUT


def create_output_dir(file_name):
    """
    Create the main output directory for a file with all subdirectories.
    
    Args:
        file_name: Name of the file
        
    Returns:
        dict: Dictionary containing all directory paths
    """
    base_dir = OUTPUT / file_name
    
    dirs = {
        'base': base_dir,
        'preprocessed': base_dir / "preprocessed",
        'masked': base_dir / "masked",
        'ocr_img': base_dir / "ocr" / "image",
        'ocr_json': base_dir / "ocr" / "json",
        'layout_img': base_dir / "layout" / "img",
        'crop': base_dir / "crop",
        'table': base_dir / "table",
        'chart': base_dir / "chart"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_cropped_img(region, crop_dir, img_name, region_num):
    """Save a cropped image region."""
    crop_path = Path(crop_dir) / f"{img_name}_region_{region_num}.png"
    cv2.imwrite(str(crop_path), region)
    print(f"[INFO] Cropped region {region_num} saved → {crop_path}")
    return crop_path


def save_table_html(table_results, img_name, region_num, table_dir):
    """Save table extraction results as HTML."""
    table_path = Path(table_dir) / f"{img_name}_table_region_{region_num}.html"
    
    try:
        for res in table_results:
            res.save_to_html(str(table_path))
        print(f"[INFO] Table HTML saved → {table_path}")
        return table_path
    except Exception as e:
        print(f"[ERROR] Failed to save table HTML → {table_path}: {e}")
        return None


def save_chart_json(chart_results, img_name, region_num, chart_dir):
    """Save chart extraction results as JSON."""
    chart_path = Path(chart_dir) / f"{img_name}_chart_region_{region_num}.json"
    
    try:        
        with open(chart_path, 'w', encoding='utf-8') as f:
            json.dump(chart_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Chart JSON saved → {chart_path}")
        return chart_path
    except Exception as e:
        print(f"[ERROR] Failed to save chart JSON → {chart_path}: {e}")
        return None


def save_layout_results(layout_results, img_name, layout_img_dir):
    """Save layout detection visualization."""
    layout_img_path = Path(layout_img_dir) / f"{img_name}_layout.png"
    
    try:
        for res in layout_results:
            res.save_to_img(str(layout_img_path))
        print(f"[INFO] Layout image saved → {layout_img_path}")
        return layout_img_path
    except TypeError as e:
        print(f"[WARNING] Skipping layout visualization for {img_name} due to format issue: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to save layout image → {layout_img_path}: {e}")
        return None


def save_ocr_results(original_ocr_results, structured_ocr_results, file_name, img_name, page_num, ocr_img_dir, ocr_json_dir):
    """Save OCR results (structured JSON and visualization images)."""
    
    # Save OCR Visualization Images
    ocr_img_path = Path(ocr_img_dir) / f"{img_name}_ocr.png"
    try:
        for res in original_ocr_results:
            res.save_to_img(str(ocr_img_path))
        print(f"[INFO] OCR visualization saved → {ocr_img_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save OCR visualization: {e}")
    
    # Save Structured OCR JSON
    ocr_json_path = Path(ocr_json_dir) / f"{file_name}_ocr.json"
    
    # Load existing JSON
    if ocr_json_path.exists():
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            try:
                all_ocr_results = json.load(f)
            except json.JSONDecodeError:
                all_ocr_results = {}
    else:
        all_ocr_results = {}
    
    # Add or update the current page
    all_ocr_results[f"page_{page_num}"] = structured_ocr_results
    
    # Save updated JSON
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(all_ocr_results, f, indent=4, ensure_ascii=False)
    
    print(f"[INFO] OCR results saved for page {page_num} → {ocr_json_path}")
    return ocr_json_path


def save_final_layout_json(layout_results, img_name, page_num, base_dir):
    """Save the final layout results with extracted content as JSON."""
    
    # Extract file_name from img_name
    if '_page_' in img_name:
        file_name = img_name.rsplit('_page_', 1)[0]
    else:
        file_name = img_name
    
    final_json_path = Path(base_dir) / f"{file_name}_all_results.json"
    
    try:
        # Load existing JSON
        if final_json_path.exists():
            with open(final_json_path, "r", encoding="utf-8") as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = {}
        else:
            all_results = {}
        
        # Prepare page data
        page_data = {
            "boxes": [],
            "metadata": {
                "image_name": img_name,
                "page_number": page_num,
                "total_boxes": 0
            }
        }
        
        for page_result in layout_results:
            boxes = page_result.get("boxes", [])
            
            # Ensure coordinates are integers
            for box in boxes:
                if 'coordinate' in box:
                    box['coordinate'] = [int(c) for c in box['coordinate']]
            
            page_data["boxes"].extend(boxes)
            page_data["metadata"]["total_boxes"] = len(boxes)
        
        # Add or update the page data
        all_results[f"page_{page_num}"] = page_data
        
        # Save updated JSON
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        
        print(f"[INFO] Complete results saved for page {page_num} → {final_json_path}")
        return final_json_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save final layout JSON for page {page_num}: {e}")
        return None


def save_results(
    dirs,
    file_name,
    img_name,
    img,
    masked_img,
    page_num,
    original_ocr_results,
    structured_ocr_results,
    layout_results,
    layout_result_save=False
):
    """
    Main function to save all pipeline results.
    
    Args:
        dirs: Dictionary of directory paths from create_output_dir()
        file_name: Name of the file
        img_name: Name of specific image/page
        img: Original/preprocessed image
        masked_img: Image with masked regions
        page_num: Page number
        original_ocr_results: Original OCR results with methods
        structured_ocr_results: Structured OCR data dictionary
        layout_results: Layout detection results
        layout_result_save: Whether to save layout visualization
    """
    base_dir = dirs['base']
    print(f"\n[SAVE] Saving results to: {base_dir}")
    
    # Save Preprocessed and Masked Images
    preprocessed_img_path = dirs['preprocessed'] / f"{img_name}_preprocessed.png"
    masked_img_path = dirs['masked'] / f"{img_name}_masked.png"
    
    cv2.imwrite(str(preprocessed_img_path), img)
    cv2.imwrite(str(masked_img_path), masked_img)
    print(f"[INFO] Preprocessed image saved → {preprocessed_img_path}")
    print(f"[INFO] Masked image saved → {masked_img_path}")
    
    # Save Layout Results
    if layout_result_save:
        save_layout_results(layout_results, img_name, dirs['layout_img'])
    
    # Save OCR Results
    save_ocr_results(
        original_ocr_results,
        structured_ocr_results,
        file_name,
        img_name,
        page_num,
        dirs['ocr_img'],
        dirs['ocr_json']
    )
    
    # Save Complete Results JSON
    save_final_layout_json(layout_results, img_name, page_num, base_dir)
    
    print(f"[SAVE] All results saved successfully\n")
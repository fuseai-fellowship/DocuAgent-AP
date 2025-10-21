import time, cv2

def mask_regions(image,image_name, layout_results):
    
    start_time = time.time()
    masked_img = image.copy()
    mask_padding=10
    
    for page_result in layout_results:
        for box in page_result["boxes"]:
            cls = box['label']
            coords = [int(x) for x in box['coordinate']]
            x1, y1, x2, y2 = coords
            
            SKIP_FOR_OCR = ["table", "chart", "image", "seal", "formula"]
            if cls in SKIP_FOR_OCR:
                # Apply padding
                x1_pad = max(0, x1 - mask_padding)
                y1_pad = max(0, y1 - mask_padding)
                x2_pad = min(image.shape[1], x2 + mask_padding)
                y2_pad = min(image.shape[0], y2 + mask_padding)
                
                # Mask region
                cv2.rectangle(masked_img, (x1_pad, y1_pad), (x2_pad, y2_pad), (255, 255, 255), thickness=-1)
                
    end_time = time.time()
    print(f"[INFO] Masking Time for {image_name} : {end_time - start_time:.2f}s")
    return masked_img


import time
import cv2


def mask_regions(image, image_name, layout_results, SKIP_FOR_OCR=None):
    """
    Mask specific regions in an image to exclude them from OCR processing.
    
    Creates a copy of the input image and masks (fills with white) regions that
    should not undergo OCR processing. This prevents OCR errors on non-text
    elements like tables, charts, and images.
    
    Args:
        image (np.ndarray): Input image in OpenCV format (BGR).
        image_name (str): Identifier/name of the image being processed (for logging).
        layout_results (list): Layout detection results containing region information.
        SKIP_FOR_OCR (list, optional): List of region types to mask. 
            Defaults to ["table", "chart", "image", "seal", "formula"].
    
    Returns:
        np.ndarray: Masked image with specified regions filled with white (255, 255, 255).
    

    Note:
        - Original image is not modified; a copy is created
        - Masked regions are filled with white color for clean OCR processing
        - Currently uses 0 padding; can be adjusted via mask_padding variable
    """
    # Set default regions to skip if not provided
    if SKIP_FOR_OCR is None:
        SKIP_FOR_OCR = ["table", "chart", "image", "seal", "formula"]
    
    # Track masking time
    start_time = time.time()
    
    # Create a copy to avoid modifying original image
    masked_img = image.copy()
    
    # Padding around masked regions (currently 0, can be adjusted)
    mask_padding = 0
    
    # Iterate through all detected layout regions
    for page_result in layout_results:
        for box in page_result["boxes"]:
            # Get region classification and coordinates
            cls = box['label']
            coords = [int(x) for x in box['coordinate']]
            x1, y1, x2, y2 = coords
            
            # Check if this region type should be masked for OCR
            if cls in SKIP_FOR_OCR:
                # Apply padding to coordinates (with boundary checks)
                x1_pad = max(0, x1 - mask_padding)
                y1_pad = max(0, y1 - mask_padding)
                x2_pad = min(image.shape[1], x2 + mask_padding)
                y2_pad = min(image.shape[0], y2 + mask_padding)
                
                # Mask region by filling with white color
                # thickness=-1 fills the entire rectangle
                cv2.rectangle(
                    masked_img, 
                    (x1_pad, y1_pad), 
                    (x2_pad, y2_pad), 
                    (255, 255, 255),  # White color
                    thickness=-1
                )
    
    end_time = time.time()
    print(f"[INFO] Masking Time for {image_name} : {end_time - start_time:.2f}s")
    
    return masked_img
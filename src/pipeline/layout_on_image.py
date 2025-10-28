
import time
from pathlib import Path
from src.models.paddle_models import layout_model


def layout_on_image(image, image_name):
    """
    Perform layout detection on a document image.
    
    Analyzes the document structure and identifies different regions (text, tables,
    charts, images, etc.). Results are sorted by vertical position (top to bottom)
    and then horizontal position (left to right) for natural reading order.
    
    Args:
        image (np.ndarray): Input image in OpenCV format (BGR).
        image_name (str): Identifier/name of the image being processed (for logging).
    
    Returns:
        list: Layout detection results containing:
            - boxes: List of detected regions with their coordinates and labels
            - Each box contains: {coordinate: [x1, y1, x2, y2], label: str, confidence: float}
    
 
    Note:
        Results are automatically sorted by position (top-to-bottom, left-to-right)
        to maintain logical reading order.
    """
    # Initialize layout detection model
    layout = layout_model()
    
    # Track detection time
    start_time = time.time()
    
    # Perform layout detection
    layout_results = layout.predict(image)
    
    # Sort detected regions by position for natural reading order
    for page_result in layout_results:
        # Each page_result["boxes"] is a list of detected regions
        boxes = page_result["boxes"]
        
        # Sort by y1 (top coordinate), then x1 (left coordinate)
        # This creates a top-to-bottom, left-to-right reading order
        boxes_sorted = sorted(boxes, key=lambda b: (b['coordinate'][1], b['coordinate'][0]))
        
        # Replace original list with sorted list
        page_result["boxes"] = boxes_sorted
    
    end_time = time.time()
    print(f"[INFO] Layout Detection Time for {image_name} : {end_time - start_time:.2f}s")
    
    return layout_results
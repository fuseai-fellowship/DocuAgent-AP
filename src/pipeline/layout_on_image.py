import time
from pathlib import Path


from src.models.paddle_models import layout_model

def layout_on_image(image, image_name):
    
    layout = layout_model()
    start_time = time.time()
    layout_results = layout.predict(image)

    for page_result in layout_results:
            # Each page_result["boxes"] is a list of regions
             boxes = page_result["boxes"]
            # Sort by y1 (top), then x1 (left)
             boxes_sorted = sorted(boxes, key=lambda b: (b['coordinate'][1], b['coordinate'][0]))
            # Replace original list with sorted list
             page_result["boxes"] = boxes_sorted
    
    end_time = time.time()
    print(f"[INFO] Layout Detection Time for {image_name} : {end_time - start_time:.2f}s")

    return layout_results
    
    


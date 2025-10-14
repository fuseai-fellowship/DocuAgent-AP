import time
from pathlib import Path


from src.models.paddle_models import layout_model

def layout_on_image(path):
    
    image_path = Path(path).resolve()
    image_name = image_path.stem
    layout = layout_model()
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    start_time = time.time()
    image_name = image_path.stem
    
    layout_results = layout.predict(str(image_path))
    layout_time = time.time()

    for page_result in layout_results:
            # Each page_result["boxes"] is a list of regions
             boxes = page_result["boxes"]
            # Sort by y1 (top), then x1 (left)
             boxes_sorted = sorted(boxes, key=lambda b: (b['coordinate'][1], b['coordinate'][0]))
            # Replace original list with sorted list
             page_result["boxes"] = boxes_sorted
    
 
    print(f"[INFO] Layout Detection Time for {image_name} : {layout_time - start_time:.2f}s")

    return layout_results
    
    


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

    print(f"[INFO] Layout Detection Time for {image_name} : {layout_time - start_time:.2f}s")

    return layout_results
    
    


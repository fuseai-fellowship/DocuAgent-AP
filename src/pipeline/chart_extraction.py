
import time
from src.models.Chart2Table import chart2table_model


def run_chart2table(image, image_name):
    """
    Extract structured table data from chart images.
    
    This function uses the Chart2Table model to analyze chart images and convert
    them into structured tabular format. The extraction is performed in batches
    and includes timing information for performance monitoring.
    
    Args:
        image (np.ndarray or PIL.Image): Input image containing the chart to be extracted.
        image_name (str): Identifier/name of the image being processed (for logging).
    
    Returns:
        list: Model prediction results containing extracted chart data in structured format.
    
    """
    
    # Initialize the chart2table model
    model = chart2table_model()
    
    print(f"[INFO] Performing Chart extraction on cropped regions for {image_name}")
    
    # Track extraction time
    start_time = time.time()
    
    # Perform chart-to-table prediction
    results = model.predict(
        input={"image": image},
        batch_size=1
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"[INFO] Chart Results extracted for {image_name} in {elapsed:.2f} seconds")
    
    return results
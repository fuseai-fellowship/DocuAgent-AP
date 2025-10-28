
import time
from src.models.Table_Recognition_V2 import table_recognition_model


def run_table_recognition(image, image_name):
    """
    Extract and recognize table structures from an image.
    
    Processes table regions in document images and converts them into structured
    formats. The function detects table cells, extracts content, and generates
    structured output with cell relationships and content.
    
    Args:
        image (np.ndarray or PIL.Image): Input image containing the table to be extracted.
        image_name (str): Identifier/name of the image being processed (for logging).
    
    Returns:
        dict or list: Table recognition results containing:
            - Structured table data (HTML, CSV, or JSON format)
            - Cell-level content and coordinates
            - Table structure information (rows, columns, merged cells)
    
    Note:
        - Document orientation classification is disabled for performance
        - Document unwarping is disabled (assumes pre-aligned images)
        - Layout detection is disabled (assumes table-only input)
        - Works best on cropped table regions from layout detection
        - Performance depends on table complexity and image quality
    """
    # Initialize table recognition model pipeline
    pipeline = table_recognition_model()
    
    print(f"[INFO] Performing Table extraction on cropped regions for {image_name}")
    
    # Track table extraction time
    start_time = time.time()
    
    # Perform table recognition with optimized settings
    # Disabled features for performance (assumes pre-processed input):
    # - use_doc_orientation_classify: Skip orientation detection
    # - use_doc_unwarping: Skip perspective correction
    # - use_layout_detection: Skip layout analysis (table-only input)
    output = pipeline.predict(
        input=image,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False
    )
    
    stop_time = time.time()
    
    print(f"[INFO] Table extraction on {image_name} took {stop_time - start_time:.2f} seconds")
    
    return output
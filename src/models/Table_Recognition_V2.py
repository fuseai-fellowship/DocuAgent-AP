
from paddleocr import TableRecognitionPipelineV2
import paddle

# Global singleton instance for the Table Recognition model
table_recognition_instance = None


def table_recognition_model():
    """
    Get or create the Table Recognition model instance (singleton pattern).
    
    Initializes the Table Recognition Pipeline V2 on first call and returns
    the same instance for subsequent calls. The model detects table structure,
    extracts cell content, and generates structured output (HTML, CSV, etc.).
    
    Returns:
        TableRecognitionPipelineV2: Initialized table recognition pipeline ready for prediction.
 
    """
    global table_recognition_instance
    
    # Check if model is already initialized
    if table_recognition_instance is None:
        print(f"[INFO] Initializing Table Recognition model")
        
        # Create and cache the table recognition pipeline
        # Preprocessing features disabled for performance optimization:
        # - use_doc_orientation_classify: Skip rotation detection
        # - use_doc_unwarping: Skip perspective correction
        table_recognition_instance = TableRecognitionPipelineV2(
            use_doc_orientation_classify=False,  # Disabled for speed
            use_doc_unwarping=False,  # Disabled for speed
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec"
        )
    
    return table_recognition_instance
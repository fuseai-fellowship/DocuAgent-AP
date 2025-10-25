
from paddlex import create_model
import paddle

# Global singleton instance for the Chart2Table model
chart2table_instance = None


def chart2table_model():
    """
    Get or create the Chart2Table model instance (singleton pattern).
    
    Initializes the PP-Chart2Table model on first call and returns the same
    instance for subsequent calls. Automatically selects GPU if available,
    otherwise falls back to CPU.
    
    Returns:
        PaddleX Model: Initialized PP-Chart2Table model instance ready for prediction.
    
  
    """
    global chart2table_instance
    
    # Check if model is already initialized
    if chart2table_instance is None:
        # Determine device availability
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)
        
        print(f"[INFO] Initializing PP-Chart2Table model on {device}...")
        
        # Create and cache the model instance
        chart2table_instance = create_model("PP-Chart2Table")
    
    return chart2table_instance
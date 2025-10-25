"""
PaddleOCR Models Module

This module provides singleton access to various PaddleOCR models including:
- Layout Detection: Document structure analysis
- OCR: Text detection and recognition
- Formula Recognition: Mathematical formula extraction

All models use lazy initialization and singleton pattern for memory efficiency.
"""

from paddleocr import LayoutDetection, PaddleOCR, FormulaRecognition
import paddle

# Global singleton instances for each model
_layout_instance = None
_ocr_instance = None
_formula_instance = None


def layout_model():
    """
    Get or create the Layout Detection model instance (singleton pattern).
    
    Initializes the PP-DocLayout model for document structure analysis on first call.
    The model detects and classifies different regions in documents such as text,
    images, tables, charts, headers, and footers.
    
    Returns:
        LayoutDetection: Initialized PP-DocLayout_plus-L model instance.
    
    Example:
        >>> model = layout_model()
        [INFO] Initializing PP-DocLayout model...
        >>> results = model.predict(image)
    
    Model Configuration:
        - model_name: PP-DocLayout_plus-L (large variant for better accuracy)
        - threshold: Custom confidence thresholds per class
            * Class 1 (image): 0.7
            * Class 2 (text): 0.2
            * Class 12 (header): 0.2
            * Class 14 (footer): 0.2
            * Class 16 (chart): 0.1
        - layout_nms: True (Non-Maximum Suppression to remove overlapping boxes)
        - layout_merge_bboxes_mode: "large" (merge strategy for overlapping regions)
        - layout_unclip_ratio: 1.1 (expand detected boxes by 10%)
    
    Note:
        - Lower thresholds for text/chart allow more detections
        - Higher threshold for images reduces false positives
        - NMS enabled to prevent duplicate detections
        - Unclip ratio helps capture full regions including margins
    """
    global _layout_instance
    
    # Check if model is already initialized
    if _layout_instance is None:
        print("[INFO] Initializing PP-DocLayout model...")
        
        # Create and cache the layout detection model
        _layout_instance = LayoutDetection(
            model_name="PP-DocLayout_plus-L",
            # Custom confidence thresholds for different region types
            threshold={
                1: 0.7,   # image: Higher threshold (fewer false positives)
                2: 0.2,   # text: Lower threshold (catch more text regions)
                12: 0.2,  # header: Lower threshold
                14: 0.2,  # footer: Lower threshold
                16: 0.1   # chart: Very low threshold (catch all charts)
            },
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
            layout_nms=True,  # Enable Non-Maximum Suppression
            layout_merge_bboxes_mode="large",  # Merge overlapping boxes (keep larger)
            layout_unclip_ratio=1.1  # Expand boxes by 10% to capture full regions
        )
    
    return _layout_instance


def ocr_model():
    """
    Get or create the OCR model instance (singleton pattern).
    
    Initializes PaddleOCR for text detection and recognition on first call.
    Uses the latest PP-OCRv5 server models optimized for English text.
    
    Returns:
        PaddleOCR: Initialized OCR model instance ready for text extraction.
    
  
    
    Note:
        - Server models provide better accuracy than mobile models
        - Document unwarping disabled due to poor performance
        - Textline orientation helps with rotated/skewed text
        - Orientation classification disabled to improve speed
    """
    global _ocr_instance
    
    # Check if model is already initialized
    if _ocr_instance is None:
        print("[INFO] Initializing PaddleOCR model...")
        
        # Create and cache the OCR model
        _ocr_instance = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",  # Latest server detection model
            text_recognition_model_name="PP-OCRv5_server_rec",  # Latest server recognition model
            lang='en',  # English language
            use_doc_orientation_classify=False,  # Disabled for speed
            use_doc_unwarping=False,  # Disabled - didn't perform well
            use_textline_orientation=True,  # Enable rotation detection
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        )
    
    return _ocr_instance


def formula_model():
    """
    Get or create the Formula Recognition model instance (singleton pattern).
    
    Initializes the Formula Recognition model for extracting mathematical
    formulas and equations from images on first call.
    
    Returns:
        FormulaRecognition: Initialized formula recognition model instance.
   
    """
    global _formula_instance
    
    # Check if model is already initialized
    if _formula_instance is None:
        print("[INFO] Initializing Formula Recognition model...")
        
        # Create and cache the formula recognition model
        _formula_instance = FormulaRecognition(
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        )
    
    return _formula_instance
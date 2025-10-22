from paddleocr import LayoutDetection, PaddleOCR, FormulaRecognition
import paddle

_layout_instance = None
_ocr_instance = None
_formula_instance = None

def layout_model():
    global _layout_instance
    if _layout_instance is None:
            print("[INFO] Initializing PP-DocLayout model...")
            _layout_instance = LayoutDetection(
                model_name="PP-DocLayout_plus-L",
                threshold={1: 0.7, 2: 0.2, 12: 0.2, 14: 0.2, 16:0.1},  # text, header, footer
                device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
                layout_nms=True,
                layout_merge_bboxes_mode="large",
                layout_unclip_ratio=1.1
            )
    return _layout_instance

def ocr_model():
    global _ocr_instance
    if _ocr_instance is None:
        print("[INFO] Initializing PaddleOCR model...")
   
        _ocr_instance = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,  # didnt perform well so not used 
            use_textline_orientation=True,
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        )
    
    return _ocr_instance

def formula_model():
    global _formula_instance
    if _formula_instance is None:
        print("[INFO] Initializing Formula Recognition model...")
        _formula_instance = FormulaRecognition(
            device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        )
    return _formula_instance
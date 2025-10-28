
from pathlib import Path
import cv2
import time
import fitz  # PyMuPDF
import numpy as np
from src.pipeline.image_pipeline import image_pipeline
from src.utils.save_results import create_output_dir


def pdf_to_images(pdf_path, dpi=150):
    """
    Convert PDF pages to images using PyMuPDF.
    
    Generator function that yields images page-by-page from a PDF document.
    Converts each page to a numpy array in OpenCV format (BGR) for further processing.
    
    Args:
        pdf_path (str or Path): Path to the input PDF file.
        dpi (int, optional): Resolution for image conversion. Defaults to 150.
            Higher DPI = better quality but larger images and slower processing.
    
    Yields:
        tuple: (page_index: int, image: np.ndarray)
            - page_index: Zero-based page number
            - image: Page rendered as BGR numpy array
    
    Raises:
        ValueError: If PDF cannot be opened or is corrupted.

    Note:
        - Uses generator pattern for memory efficiency with large PDFs
        - Automatically handles RGBA and RGB color spaces
        - Always converts to BGR for OpenCV compatibility
    """
    pdf_name = Path(pdf_path).stem
    
    # Open PDF document
    try:
        doc = fitz.open(pdf_path)
        print("[INFO] PDF opened successfully.")
    except Exception as e:
        raise ValueError(f"Error opening PDF: {e}")
    
    print(f"[INFO] Converting PDF to images for {pdf_name}")
    start_time = time.time()
    
    try:
        # Process each page
        for i, page in enumerate(doc):
            # Render page to pixmap at specified DPI
            pix = page.get_pixmap(dpi=dpi)
            
            # Convert pixmap to numpy array in OpenCV format
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Handle different color spaces
            if pix.n == 4:  # RGBA format
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB format
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            yield i, img
            
    except Exception as e:
        print(f"[ERROR] Failed to convert PDF to images: {e}")
    finally:
        # Always close the document to free resources
        doc.close()
        end_time = time.time()
        print(f"[INFO] PDF to image conversion took {end_time - start_time:.2f} seconds")


def process_pdf_pipeline(pdf_path, preprocess=False, layout_result_save=False, dpi=150):
    """
    Process an entire PDF document through the analysis pipeline.
    
    Main orchestration function that converts a PDF to images and processes each page
    through the complete document analysis pipeline including layout detection, OCR,
    table extraction, and chart extraction.
    
    Args:
        pdf_path (str or Path): Path to the input PDF file.
        preprocess (bool, optional): Whether to apply image preprocessing. Defaults to False.
        layout_result_save (bool, optional): Whether to save layout detection visualizations. 
            Defaults to False.
        dpi (int, optional): Resolution for PDF-to-image conversion. Defaults to 150.
    
    Returns:
        None: Results are saved to output directory structure.
    
   
    Note:
        - Processes pages sequentially (not parallel)
        - Creates output directories automatically
        - Includes detailed timing information for performance monitoring
        - Handles errors gracefully without stopping entire pipeline
    """
    pdf_name = Path(pdf_path).stem
    
    # Create output directory structure
    dirs = create_output_dir(pdf_name)
    
    # Track total processing time
    start_time = time.time()
    
    try:
        # Process each page of the PDF
        for i, img in pdf_to_images(pdf_path, dpi=dpi):
            page_start_time = time.time()
            print(f"[INFO] Processing page {i+1} of {pdf_name}")
            
            # Run the image through the complete pipeline
            image_pipeline(
                img=img,
                pdf_name=pdf_name,
                preprocess=preprocess,
                layout_result_save=layout_result_save,
                pdf_page_num=i + 1,
                dirs=dirs
            )
            
            page_end_time = time.time()
            print(f"[INFO] Processing page {i+1} took {page_end_time - page_start_time:.2f} seconds")
            
    finally:
        # Always print total time, even if error occurs
        end_time = time.time()
        print(f"[INFO] Pipeline on {pdf_name} took {end_time - start_time:.2f} seconds")
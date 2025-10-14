
from pathlib import Path
import fitz

from src.utils.paths import RAW
def pdf_to_images(pdf_path):
    pdf_name = Path(pdf_path).stem
   
    save_dir = RAW / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    print("[INFO] PDF opened successfully.")

    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  
        try:
            img_path = save_dir / f"page_{i+1}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"Error converting pdf to images in page {i+1}: {e}")
    doc.close()
    return image_paths

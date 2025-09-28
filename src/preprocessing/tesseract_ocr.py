from PIL import Image
import pytesseract
from image_preprocessing import preprocess_document
import cv2
import numpy as np

image = Image.open("temp/processed_image.jpg")
text = pytesseract.image_to_data(image, lang='eng', output_type='data.frame')
# print(text)

# bounded_box = pytesseract.image_to_boxes("temp/processed_image.jpg")
# https://stackoverflow.com/questions/55406993/how-to-get-confidence-of-each-line-using-pytesseract

def pdf_to_images(pdf_path, pdf_name) -> list:
    save_dir = RAW / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  
        img_path = save_dir / f"page_{i+1}.png"
        pix.save(str(img_path))
        image_paths.append(str(img_path))

    return image_paths

def get_confidence_score(text):
    text = text[text.conf != -1]
    # lines = text.groupby('block_num')['text'].apply(list)

    # the confidence scoring
    conf = text.groupby(['block_num'])['conf'].mean()
    return conf

def process_image(image_path):
    image = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_image = preprocess_document(cv_image)
    cv2.imwrite("outputs/output3.png", processed_image)
    text = pytesseract.image_to_data(processed_image, lang='eng', output_type='data.frame')
    lines = text.groupby('line_num')['text'].apply(lambda x: " ".join(x.dropna().astype(str)))

    print("\n--- OCR Output ---")
    for line in lines:
        if line.strip():
            print(line)

image_path = "data/sample_image3.jpg"
process_image(image_path)
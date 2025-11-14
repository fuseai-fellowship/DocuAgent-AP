from src.pipeline.image_pipeline import image_pipeline
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.utils.paths import DATA, OUTPUT
from src.pipeline.chat_with_LLM import chat_with_gemini, chat_with_openai
from pathlib import Path
from datasets import load_dataset

# Load FUNSD dataset
funsd = load_dataset("nielsr/funsd")
test_set = funsd['test']
train_set = funsd['train']

# Temporary directory for saving images (currently commented out)
# TEMP_DIR = Path("temp_funsd_images")
# TEMP_DIR.mkdir(exist_ok=True)

# Save first 50 test images to TEMP_DIR (commented out)
# for i in range(50):
#     example = test_set[i]
#     img = example['image']
#     img_path = TEMP_DIR / f"{example['id']}.png"
#     img.save(img_path)

# Save all training images to TEMP_DIR with "_train" suffix (commented out)
# for i in range(len(train_set)):
#     example = train_set[i]
#     img = example['image']
#     img_path = TEMP_DIR / f"{i}_train.png"
#     img.save(img_path)

if __name__ == "__main__":
    # -----------------------------
    # Example: Running pipeline on FUNSD training set
    # -----------------------------
    # for i in range(len(train_set)):
    #     example = train_set[i]
    #     img_path = TEMP_DIR / f"{example['id']}_train.png"
    #     print(f"Processing FUNSD train image {i+1}: {img_path.name}")
    #
    #     # Run the document processing pipeline
    #     image_pipeline(img_path=img_path, preprocess=True, layout_result_save=False)

    # -----------------------------
    # Example: Running pipeline on a single image
    # -----------------------------
    sample_chart = DATA / "example.png"
    # image_pipeline(img_path=sample_chart, preprocess=False, layout_result_save=True)
    chat_with_openai(json_file_path='outputs/resume_upahar_khatiwada/resume_upahar_khatiwada_all_results.json', save_markdown=False)


    # sample_pdf4 = DATA / "sample_pdf1.pdf"
    # process_pdf_pipeline(pdf_path=sample_pdf4, preprocess=False, layout_result_save=True)
    # print(f"OCR results for image and PDF are done")

    # Run pipeline on additional sample images
    # sample_img1 = DATA / "sample_image1.png"
    # image_pipeline(img_path=sample_img1, preprocess=False, layout_result_save=True)
    # print("OCR done for sample_image1.png")
    #
    # sample_img2 = DATA / "sample_image2.png"
    # image_pipeline(img_path=sample_img2, preprocess=False, layout_result_save=True)
    # print("OCR done for sample_image2.png")
    #
    # sample_img4 = DATA / "sample_image4.png"
    # image_pipeline(img_path=sample_img4, preprocess=False, layout_result_save=True)
    #
    # sample_img5 = DATA / "test.png"
    # image_pipeline(img_path=sample_img5, preprocess=False, layout_result_save=True)
    # print("OCR done for test.png")
    # chat_with_gemini(OUTPUT / "test" / "test_all_results.json")
    #
    # sample_pdf2 = DATA / "test_pdf.pdf"
    # process_pdf_pipeline(pdf_path=sample_pdf2, layout_result_save=True, preprocess=False)
    # print("OCR done for test_pdf.pdf")
    # chat_with_gemini(OUTPUT / "test_pdf" / "test_pdf_all_results.json")
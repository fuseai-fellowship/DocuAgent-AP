from src.models.Table_Recognition_V2 import table_recognition_model
import time

def run_table_recognition(image_path, output_path):
    pipeline = table_recognition_model()
    print(f"[INFO] Performing Table extraction on cropped regions for {image_path}")
    start_time = time.time()
    output = pipeline.predict(
        input=image_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
    )

    for res in output:
        # res.save_to_img("./output/") 
        # res.save_to_xlsx("./output/") 
        res.save_to_html(output_path)
        # res.save_to_json("./output/")
    
    # return output

    stop_time = time.time()
    print(f"[INFO] Table Results saved to {output_path} in {(start_time-stop_time):.2f} seconds")
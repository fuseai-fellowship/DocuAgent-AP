from paddlex import create_pipeline
import paddle

table_recognition_instance = None

def table_recognition_model():
    global table_recognition_instance

    if table_recognition_instance is None:
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

        print(f"[INFO] Initializing PaddleX Table Recognition model on {device}...")
        table_recognition_instance = create_pipeline(pipeline="table_recognition_v2")

    return table_recognition_instance

def run_table_recognition(image_path, output_path):
    pipeline = table_recognition_model()

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
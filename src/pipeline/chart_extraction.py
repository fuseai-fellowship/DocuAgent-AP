from src.models.Chart2Table import chart2table_model
import time

def run_chart2table(image_path, output_path="./output"):
    model = chart2table_model()
    print(f"[INFO] Performing Chart extraction on cropped regions for {image_path}")
    start_time = time.time()
    results = model.predict(
        input={"image": image_path},
        batch_size=1
    )

    for res in results:
        res.print()
        res.save_to_json(output_path)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] Chart Results saved to {output_path} in {elapsed:.2f} seconds")
    return results
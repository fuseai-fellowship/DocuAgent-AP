from paddlex import create_model
import paddle

chart2table_instance = None

def chart2table_model():
    global chart2table_instance

    if chart2table_instance is None:
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

        print(f"[INFO] Initializing PP-Chart2Table model on {device}...")
        print("Paddle compiled with CUDA:", paddle.is_compiled_with_cuda())
        print("Current device:", paddle.get_device())

        chart2table_instance = create_model("PP-Chart2Table")

    return chart2table_instance
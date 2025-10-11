from paddlex import create_model
import paddle

paddle.set_device('gpu')
print("Paddle compiled with CUDA:", paddle.is_compiled_with_cuda())
print("Current device:", paddle.get_device())

model = create_model('PP-Chart2Table')
results = model.predict(
    input={"image": "test8.png"}, # placeholder for input document
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res8.json") # output directory
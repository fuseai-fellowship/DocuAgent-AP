from paddlex import create_pipeline
import paddle

paddle.set_device('gpu')
pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="latex2.jpg", # placeholder for input document
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print() # output in different format
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
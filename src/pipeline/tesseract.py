from PIL import Image
import pytesseract

image = Image.open("temp/processed_image.jpg")

text = pytesseract.image_to_data(image, lang='eng', output_type='data.frame')
print(text)

# bounded_box = pytesseract.image_to_boxes("temp/processed_image.jpg")

# https://stackoverflow.com/questions/55406993/how-to-get-confidence-of-each-line-using-pytesseract

text = text[text.conf != -1]
lines = text.groupby('block_num')['text'].apply(list)

# the confidence scoring
conf = text.groupby(['block_num'])['conf'].mean()
print(conf)
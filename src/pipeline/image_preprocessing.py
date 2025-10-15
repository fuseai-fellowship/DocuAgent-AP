import cv2
import matplotlib.pyplot as plt
import numpy as np

# Display image
def display(image):
    if isinstance(image, str):
        im_data = plt.imread(image)
    else:
        im_data = image

    dpi = 80
    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis('off')

    if len(im_data.shape) == 2:
        ax.imshow(im_data, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB))
    plt.show()


# Invert image
def invert_image(image):
    return cv2.bitwise_not(image)

# Binarization (handles both color and grayscale input)
def binarization(image):
    if len(image.shape) == 3:  # color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # already grayscale
        gray = image
    return gray


# Noise removal
def noise_removal(image):
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)
    return image

# Thin font
def thin_font(image):
    image = invert_image(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = invert_image(image)
    return image

# Thick font
def thick_font(image):
    image = invert_image(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = invert_image(image)
    return image

# Sharpen image for blurred text
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Apply CLAHE for contrast enhancement
def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# Deskew helper functions
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

def deskew(image):
    angle = skew_angle(image)
    rotated = rotate_bound(image, angle)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h_proj = np.sum(bw == 0, axis=1)
    v_proj = np.sum(bw == 0, axis=0)
    if np.std(v_proj) > np.std(h_proj):
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        angle += 90
    gray_final = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    h_mid = gray_final.shape[0] // 2
    top_sum = np.sum(gray_final[:h_mid])
    bottom_sum = np.sum(gray_final[h_mid:])
    if bottom_sum > top_sum:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        angle += 180
    angle = (angle + 180) % 360 - 180
    return rotated, angle


# function to remove borders
def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]
# Full preprocessing pipeline 
def preprocess_document(image):
    image, _ = deskew(image)
    image = remove_borders(image)
    image = sharpen_image(image)
    image = enhance_contrast(image)
    image = binarization(image)
    image = noise_removal(image)
    return image

img = cv2.imread("data/skewed_image.jpg")
processed = preprocess_document(img)
# display(processed)
cv2.imwrite("temp/processed_image.jpg", processed)
from PIL import Image
import pyccv

filepath = './image_for_text/sloneczniki.jpg'
img = Image.open(filepath)
size = 240  # Normalize image size
threshold = 25
ccv = pyccv.calc_ccv(img, size, threshold)
print(ccv.shape)  # 128
print(ccv)

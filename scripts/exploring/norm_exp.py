import cv2

img_jeep = cv2.imread('./image_for_text/lincoln.jpg')
print("Image data before Normalize:", img_jeep)

# Normalize the image
img_normalized = cv2.normalize(img_jeep, 1, None, 255,
                               cv2.NORM_MINMAX)

cv2.imshow('Original Iamge', img_jeep)
cv2.imshow('Normalized Image', img_normalized)
max_o = max(img_jeep.flatten())
max_n = max(img_normalized.flatten())
print(max_o)
print(max_n)
print(img_jeep)
print(img_normalized)
cv2.waitKey(0)

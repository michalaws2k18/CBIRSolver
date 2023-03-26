import cv2
from matplotlib import pyplot as plt
import numpy as np

img_jeep = cv2.imread('./image_for_text/lincoln.jpg')
print("Image data before Normalize:", img_jeep)

# Normalize the image
img_normalized = cv2.normalize(img_jeep, 1, None, 255,
                               cv2.NORM_MINMAX)

# cv2.imshow('Original Iamge', img_jeep)
# cv2.imshow('Normalized Image', img_normalized)
# max_o = max(img_jeep.flatten())
# max_n = max(img_normalized.flatten())
# print(max_o)
# print(max_n)
# print(img_jeep)
# print(img_normalized)
# cv2.waitKey(0)

# gray image normalize
img = cv2.imread('./image_for_text/lena.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray, 0, None, 255,
                          cv2.NORM_MINMAX)

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(wspace=0.6)
axs[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
axs[0, 0].axis("off")
axs[0, 0].set_title('Obraz czarno-biały oryginalny')
axs[0, 1].set_title('Obraz czarno-biały znormalizowany')

axs[0, 1].imshow(gray_norm, cmap='gray', vmin=0, vmax=255)
axs[0, 1].axis("off")

axs[1, 0].hist(gray[:, :].flatten(), bins=256, color='black')
axs[1, 1].hist(gray_norm[:, :].flatten(), bins=256, color='black')
axs[1, 0].set_xlabel("jasność pikseli")
axs[1, 0].set_ylabel("liczność pikseli")
axs[1, 1].set_xlabel("jasność pikseli")
axs[1, 1].set_ylabel("liczność pikseli")
axs[1, 0].set_xlim([0, 256])
axs[1, 1].set_xlim([0, 256])

max_o = max(gray.flatten())
max_n = max(gray_norm.flatten())

min_o = min(gray.flatten())
min_n = min(gray_norm.flatten())

print(max_o)
print(max_n)

print(min_o)
print(min_n)


# For RGB image
print(img.shape)

norm_var = np.zeros((512, 512))

img_norm = cv2.normalize(img, norm_var, 0, 255,
                         cv2.NORM_MINMAX)


fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(wspace=0.6)
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].axis("off")
axs[0, 0].set_title('Obraz oryginalny')
axs[0, 1].set_title('Obraz znormalizowany')

axs[0, 1].imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
axs[0, 1].axis("off")

# axs[1, 0].hist(img[:, :, 0].flatten(), bins=256, color='blue')
# axs[1, 0].hist(img[:, :, 1].flatten(), bins=256, color='green')
axs[1, 0].hist(img[:, :, 2].flatten(), bins=256, color='red')

# axs[1, 1].hist(img_norm[:, :, 0].flatten(), bins=256, color='blue')
# axs[1, 1].hist(img_norm[:, :, 1].flatten(), bins=256, color='green')
axs[1, 1].hist(img_norm[:, :, 2].flatten(), bins=256, color='red')
axs[1, 0].set_xlabel("jasność pikseli")
axs[1, 0].set_ylabel("liczność pikseli")
axs[1, 1].set_xlabel("jasność pikseli")
axs[1, 1].set_ylabel("liczność pikseli")
axs[1, 0].set_xlim([0, 256])
axs[1, 1].set_xlim([0, 256])

max_o = max(img[:, :, 1].flatten())
max_n = max(img_norm[:, :, 1].flatten())

min_o = min(img[:, :, 2].flatten())
min_n = min(img_norm[:, :, 2].flatten())

print(max_o)
print(max_n)

print(min_o)
print(min_n)
plt.show()

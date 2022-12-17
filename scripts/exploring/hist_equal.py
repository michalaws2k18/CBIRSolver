import cv2
import numpy as np
from config import EXAMPLE_IMG_PATH
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import statistics
from PIL import Image

img = cv2.imread('./image_for_text/sloneczniki.jpg')
ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# equalize the histogram of the Y channel
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

# convert back to RGB color-space from YCrCb
equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

# cv2.imshow('equalized_img', equalized_img)
# cv2.waitKey(0)

img_grey = cv2.imread('./image_for_text/sloneczniki.jpg', 0)
print(img_grey)
img_grey_equal = cv2.equalizeHist(img_grey)


# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE()
img_grey_equal_clahe = clahe.apply(img_grey)

fig, axs = plt.subplots(2, 3)
fig.subplots_adjust(wspace=0.5)
axs[0, 0].imshow(img_grey, cmap='gray', vmin=0, vmax=255)
axs[0, 0].axis("off")
axs[0, 0].set_title('Obraz czarno-biały')
axs[0, 1].set_title('Wyrównanie tradycyjne')
axs[0, 2].set_title('Wyrównanie CLAHE')

axs[0, 1].imshow(img_grey_equal, cmap='gray', vmin=0, vmax=255)
axs[0, 1].axis("off")

axs[0, 2].imshow(img_grey_equal_clahe, cmap='gray', vmin=0, vmax=255)
axs[0, 2].axis("off")

axs[1, 0].set_ylim([0, 13000])
axs[1, 1].set_ylim([0, 13000])
axs[1, 2].set_ylim([0, 13000])

# axs[1, 0].grid()
# axs[1, 1].grid()
# axs[1, 2].grid()

axs[1, 0].hist(img_grey[:, :].flatten(), bins=256, color='green')

axs[1, 1].hist(img_grey_equal[:, :].flatten(), bins=256, color='blue')

axs[1, 2].hist(img_grey_equal_clahe[:, :].flatten(), bins=256, color='red')


# axs[1, 2].hist(sl_bt, bins=256, color='blue')
# axs[2, 2].hist(sl_bt, bins=number_of_bins, color='blue')
plt.show()

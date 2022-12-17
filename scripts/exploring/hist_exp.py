import cv2
import numpy as np
from config import EXAMPLE_IMG_PATH
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import statistics
from PIL import Image
# import pyvips

img = cv2.imread('./image_for_text/sloneczniki.jpg')
# cv2.imshow('original image', img)
# cv2.waitKey(0)

img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r, b, g = cv2.split(img_show)

# set green and red channels to 0
rt = img_show.copy()
rt[:, :, 1] = 0
rt[:, :, 2] = 0

# set red and blue channels to 0
gt = img_show.copy()
gt[:, :, 0] = 0
gt[:, :, 2] = 0

# set red and green channels to 0
bt = img_show.copy()
bt[:, :, 0] = 0
bt[:, :, 1] = 0

# print(hsv_img)
# cv2.imshow('HSV image', hsv_img)
# cv2.waitKey(0)

number_of_bins = 32
# histH = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
# plt.figure()
# plt.title("Hue value histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(histH)
# plt.xlim([0, 256])
# # plt.show()

# histH32 = cv2.calcHist([hsv_img], [0], None, [number_of_bins], [0, 256])
# plt.figure()
# plt.title("Hue value histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(histH32)
# plt.xlim([0, 32])
# # plt.show()

# # histG = cv2.calcHist([hsv_img], [1], None, [number_of_bins], [0, 256])
# # histR = cv2.calcHist([hsv_img], [2], None, [number_of_bins], [0, 256])

# # histdata = np.row_stack((histB, histG, histR))

# h = hsv_img.copy()
# # set green and red channels to 0
# h[:, :, 1] = 0
# h[:, :, 2] = 0


# s = hsv_img.copy()
# # set blue and red channels to 0
# s[:, :, 0] = 0
# s[:, :, 2] = 0

# v = hsv_img.copy()
# # set blue and green channels to 0
# v[:, :, 0] = 0
# v[:, :, 1] = 0

# print(type(h))
# print(np.shape(h))
# sliced_h = h[:, :, 0].flatten()
# print(np.shape(sliced_h))
# plt.figure()
# plt.hist(sliced_h, bins=32)

sl_rt = rt[:, :, 0].flatten()
sl_gt = gt[:, :, 1].flatten()
sl_bt = bt[:, :, 2].flatten()


fig, axs = plt.subplots(3, 3)
fig.subplots_adjust(wspace=0.5)
axs[0, 0].imshow(rt)
axs[0, 0].axis("off")
# axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].imshow(gt)
axs[0, 1].axis("off")
axs[0, 2].imshow(bt)
axs[0, 2].axis("off")

axs[1, 0].hist(sl_rt, bins=256, color='red')
axs[2, 0].hist(sl_rt, bins=number_of_bins, color='red')

axs[1, 1].hist(sl_gt, bins=256, color='green')
axs[2, 1].hist(sl_gt, bins=number_of_bins, color='green')

axs[1, 2].hist(sl_bt, bins=256, color='blue')
axs[2, 2].hist(sl_bt, bins=number_of_bins, color='blue')

# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].imshow(grey_img, cmap=plt.cm.binary)
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')

# plt.show()
# print(b)
# cv2.normalize(r, b, alpha=0, norm_type=cv2.NORM_MINMAX)
# print(b)
# pyvips.hist_exual(img, img)


img_jeep = cv2.imread('./image_for_text/jeep.jpg')
print("Image data before Normalize:", img_jeep)

# Normalize the image
img_normalized = cv2.normalize(img_jeep, 1, None, 255,
                               cv2.NORM_MINMAX)

r = img_jeep.copy()
r[:, :, 0] = 0
r[:, :, 1] = 0

rn = img_normalized.copy()
rn[:, :, 0] = 0
rn[:, :, 1] = 0

b = img_jeep.copy()
b[:, :, 1] = 0
b[:, :, 2] = 0

bn = img_normalized.copy()
bn[:, :, 1] = 0
bn[:, :, 2] = 0

g = img_jeep.copy()
g[:, :, 0] = 0
g[:, :, 2] = 0

gn = img_normalized.copy()
gn[:, :, 0] = 0
gn[:, :, 2] = 0

fig, axs = plt.subplots(2, 3)
fig.subplots_adjust(wspace=0.5)

axs[0, 0].hist(r[:, :, 2].flatten(), bins=256, color='red')
axs[1, 0].hist(rn[:, :, 2].flatten(), bins=256, color='red')

axs[0, 1].hist(b[:, :, 0].flatten(), bins=256, color='green')
axs[1, 1].hist(bn[:, :, 0].flatten(), bins=256, color='green')

axs[0, 2].hist(g[:, :, 1].flatten(), bins=256, color='blue')
axs[1, 2].hist(gn[:, :, 1].flatten(), bins=256, color='blue')

# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].imshow(grey_img, cmap=plt.cm.binary)
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')

plt.show()

# visualize the normalized image
cv2.imshow('Normalized Image', img_normalized)
cv2.waitKey(0)

print("Image data after Normalize:", img_normalized)

equ = cv2.equalizeHist(img_jeep)
img_equal = np.hstack((img_jeep, equ))
cv2.imwrite('res.png', img_equal)
cv2.imshow('Equalised Image', img_equal)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_img = cv2.imread('./image_for_text/jeep.jpg')

# convert from RGB color-space to YCrCb
ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

# equalize the histogram of the Y channel
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

# convert back to RGB color-space from YCrCb
equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

cv2.imshow('equalized_img', equalized_img)
cv2.waitKey(0)

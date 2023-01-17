import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
from textwrap import wrap
import numpy as np

img = cv2.imread('./image_for_text/sloneczniki.jpg')
ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ycrcb_img_stan = deepcopy(ycrcb_img)
hsv_img_stan = deepcopy(hsv_img)
# equalize the histogram of the Y channel
ycrcb_img_stan[:, :, 0] = cv2.equalizeHist(ycrcb_img_stan[:, :, 0])
hsv_img_stan[:, :, 2] = cv2.equalizeHist(hsv_img_stan[:, :, 2])

# convert back to RGB color-space from YCrCb
equalized_img_y_stan = cv2.cvtColor(ycrcb_img_stan, cv2.COLOR_YCrCb2BGR)
equalized_img_v_stan = cv2.cvtColor(hsv_img_stan, cv2.COLOR_HSV2BGR)

fig, axs = plt.subplots(1, 3)
fig.subplots_adjust(wspace=0.75)
axs[0].axis("off")
axs[1].axis("off")
axs[2].axis("off")

titlewidth = 20
axs[0].set_title("\n".join(wrap("Obraz oryginalany", titlewidth)))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1].set_title("\n".join(wrap("Składowa Y w przestrzeni YCbCr", titlewidth)))
axs[1].imshow(ycrcb_img[:, :, 0], cmap='gray', vmin=0, vmax=255)
axs[2].set_title("\n".join(wrap("Składowa V w przestrzeni HSV", titlewidth)))
axs[2].imshow(hsv_img[:, :, 2], cmap='gray', vmin=0, vmax=255)


# img_grey = cv2.imread('./image_for_text/sloneczniki.jpg', 0)
# print(img_grey)
# img_grey_equal = cv2.equalizeHist(img_grey)


# # create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE()
ycrcb_img_clahe = deepcopy(ycrcb_img)
ycrcb_img_clahe[:, :, 0] = clahe.apply(ycrcb_img_clahe[:, :, 0])
equalized_img_y_clahe = cv2.cvtColor(ycrcb_img_clahe, cv2.COLOR_YCrCb2BGR)
# img_grey_equal_clahe = clahe.apply(img_grey)

fig, axs = plt.subplots(2, 3)
fig.suptitle("Wyrównanie histogramu w przestrzeni YCbCr")
fig.subplots_adjust(wspace=0.75)
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].axis("off")

axs[0, 1].imshow(cv2.cvtColor(equalized_img_y_stan, cv2.COLOR_BGR2RGB))
axs[0, 1].axis("off")

axs[0, 2].imshow(cv2.cvtColor(equalized_img_y_clahe, cv2.COLOR_BGR2RGB))
axs[0, 2].axis("off")

axs[0, 0].set_title('Obraz oryginalny')
axs[0, 1].set_title('Wyrównanie tradycyjne')
axs[0, 2].set_title('Wyrównanie CLAHE')

# axs[0, 1].imshow(img_grey_equal, cmap='gray', vmin=0, vmax=255)
# axs[0, 1].axis("off")

# axs[0, 2].imshow(img_grey_equal_clahe, cmap='gray', vmin=0, vmax=255)
# axs[0, 2].axis("off")

axs[1, 0].set_ylim([0, 15000])
axs[1, 1].set_ylim([0, 15000])
axs[1, 2].set_ylim([0, 15000])

axs[1, 0].set_xlim([0, 256])
axs[1, 1].set_xlim([0, 256])
axs[1, 2].set_xlim([0, 256])

# # axs[1, 0].grid()
# # axs[1, 1].grid()
# # axs[1, 2].grid()

axs[1, 0].hist(img[:, :, 0].flatten(), bins=256, color='blue')
axs[1, 0].hist(img[:, :, 1].flatten(), bins=256, color='green')
axs[1, 0].hist(img[:, :, 2].flatten(), bins=256, color='red')
axs[1, 0].set_xlabel("jasność pikseli")
axs[1, 0].set_ylabel("liczność pikseli")


axs[1, 1].hist(equalized_img_y_stan[:, :, 0].flatten(), bins=256, color='blue')
axs[1, 1].hist(equalized_img_y_stan[:, :, 1].flatten(),
               bins=256, color='green')
axs[1, 1].hist(equalized_img_y_stan[:, :, 2].flatten(), bins=256, color='red')
axs[1, 1].set_xlabel("jasność pikseli")
axs[1, 1].set_ylabel("liczność pikseli")


axs[1, 2].hist(equalized_img_y_clahe[:, :, 0].flatten(),
               bins=256, color='blue')
axs[1, 2].hist(equalized_img_y_clahe[:, :, 1].flatten(),
               bins=256, color='green')
axs[1, 2].hist(equalized_img_y_clahe[:, :, 2].flatten(), bins=256, color='red')
axs[1, 2].set_xlabel("jasność pikseli")
axs[1, 2].set_ylabel("liczność pikseli")


hsv_img_clahe = deepcopy(hsv_img)
hsv_img_clahe[:, :, 2] = clahe.apply(hsv_img_clahe[:, :, 2])
equalized_img_v_clahe = cv2.cvtColor(hsv_img_clahe, cv2.COLOR_HSV2BGR)

# w opraviu o wartośc V z HSV
fig, axs = plt.subplots(2, 3)
fig.suptitle("Wyrównanie histogramu w przestrzeni HSV")
fig.subplots_adjust(wspace=0.75)
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].axis("off")

axs[0, 1].imshow(cv2.cvtColor(equalized_img_v_stan, cv2.COLOR_BGR2RGB))
axs[0, 1].axis("off")

axs[0, 2].imshow(cv2.cvtColor(equalized_img_v_clahe, cv2.COLOR_BGR2RGB))
axs[0, 2].axis("off")

axs[0, 0].set_title('Obraz oryginalny')
axs[0, 1].set_title('Wyrównanie tradycyjne')
axs[0, 2].set_title('Wyrównanie CLAHE')


axs[1, 0].set_ylim([0, 15000])
axs[1, 1].set_ylim([0, 15000])
axs[1, 2].set_ylim([0, 15000])

axs[1, 0].set_xlim([0, 256])
axs[1, 1].set_xlim([0, 256])
axs[1, 2].set_xlim([0, 256])


axs[1, 0].hist(img[:, :, 0].flatten(), bins=256, color='blue')
axs[1, 0].hist(img[:, :, 1].flatten(), bins=256, color='green')
axs[1, 0].hist(img[:, :, 2].flatten(), bins=256, color='red')
axs[1, 0].set_xlabel("jasność pikseli")
axs[1, 0].set_ylabel("liczność pikseli")


axs[1, 1].hist(equalized_img_v_stan[:, :, 0].flatten(), bins=256, color='blue')
axs[1, 1].hist(equalized_img_v_stan[:, :, 1].flatten(),
               bins=256, color='green')
axs[1, 1].hist(equalized_img_v_stan[:, :, 2].flatten(), bins=256, color='red')
axs[1, 1].set_xlabel("jasność pikseli")
axs[1, 1].set_ylabel("liczność pikseli")

axs[1, 2].hist(equalized_img_v_clahe[:, :, 0].flatten(),
               bins=256, color='blue')
axs[1, 2].hist(equalized_img_v_clahe[:, :, 1].flatten(),
               bins=256, color='green')
axs[1, 2].hist(equalized_img_v_clahe[:, :, 2].flatten(), bins=256, color='red')
axs[1, 2].set_xlabel("jasność pikseli")
axs[1, 2].set_ylabel("liczność pikseli")


plt.show()

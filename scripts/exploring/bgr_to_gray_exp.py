import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./image_for_text/lena.png')


def convertLighthness(image_bgr):
    rows, cols, dim = image_bgr.shape
    rows = int(rows)
    cols = int(cols)
    gray = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            val = int((max(image_bgr[i, j, :])+min(image_bgr[i, j, :]))/2)
            gray[i, j] = val
    return gray


def convertMean(image_bgr):
    rows, cols, dim = image_bgr.shape
    rows = int(rows)
    cols = int(cols)
    gray = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            val = int(np.mean(image_bgr[i, j, :]))
            gray[i, j] = val
    return gray


def convertCoef(image_bgr, b, g, r):
    rows, cols, dim = image_bgr.shape
    rows = int(rows)
    cols = int(cols)
    gray = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            val = int(b*image_bgr[i, j, 0]+g *
                      image_bgr[i, j, 1]+r*image_bgr[i, j, 2])
            gray[i, j] = val
    return gray


gray_lightness = convertLighthness(img)
gray_mean = convertMean(img)
gray_coef1 = convertCoef(img, 0.07, 0.72, 0.21)

fig, axs = plt.subplots(5, 1)
fig.subplots_adjust(hspace=0.04)
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].axis("off")
# axs[0].set_title('Obraz oryginalny RGB')
# axs[1].set_title('konwersja jasność')

axs[1].imshow(gray_lightness, cmap='gray', vmin=0, vmax=255)
axs[1].axis("off")

axs[2].imshow(gray_mean, cmap='gray', vmin=0, vmax=255)
axs[2].axis("off")
# axs[2].set_title('konwersja średnia')

axs[3].imshow(gray_coef1, cmap='gray', vmin=0, vmax=255)
axs[3].axis("off")
# axs[3].set_title('konwersja średnia')
axs[4].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
              cmap='gray', vmin=0, vmax=255)
axs[4].axis("off")
plt.show()

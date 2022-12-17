import numpy as np
import cv2


def calculateHistogram(number_of_bins, image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histB = cv2.calcHist([hsv_img], [0], None, [number_of_bins], [0, 256])
    histG = cv2.calcHist([hsv_img], [1], None, [number_of_bins], [0, 256])
    histR = cv2.calcHist([hsv_img], [2], None, [number_of_bins], [0, 256])

    histdata = np.row_stack((histB, histG, histR))
    return histdata


def calculatesaveHistogram(number_of_bins, image):
    img = cv2.imread(image)
    histdata = calculateHistogram(number_of_bins, img)
    # print(histdata)
    saveas = image[:image.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)

def calculatesaveHistogram2(number_of_bins, image):
    img = runHistEqual(image)
    histdata = calculateHistogram(number_of_bins, img)
    # print(histdata)
    saveas = image[:image.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)

def normaliseHistogram(canal_hist, ):
    cv2.normalize(canal_hist, canal_hist, alpha=0, norm_type=cv2.NORM_MINMAX)


def runHistEqual(image_path):
    rgb_img = cv2.imread(image_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

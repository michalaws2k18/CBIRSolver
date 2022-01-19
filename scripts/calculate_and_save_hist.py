import numpy as np
import cv2


def calculateHistogram(number_of_bins, image):
    img = cv2.imread(image)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histB = cv2.calcHist([hsv_img], [0], None, [number_of_bins], [0, 256])
    histG = cv2.calcHist([hsv_img], [1], None, [number_of_bins], [0, 256])
    histR = cv2.calcHist([hsv_img], [2], None, [number_of_bins], [0, 256])

    histdata = np.row_stack((histB, histG, histR))
    return histdata


def calculatesaveHistogram(number_of_bins, image):
    histdata = calculateHistogram(number_of_bins, image)
    # print(histdata)
    saveas = image[:image.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)

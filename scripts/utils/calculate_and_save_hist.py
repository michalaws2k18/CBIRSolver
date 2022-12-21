import numpy as np
import cv2


def calculateHistogram(number_of_bins, image, conversion=cv2.COLOR_BGR2HSV):
    hsv_img = cv2.cvtColor(image, conversion)
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


def calculatesaveHistogram2(number_of_bins, image_path, equal_alg):
    img = runHistEqual(image_path, equal_alg)
    histdata = calculateHistogram(number_of_bins, img, cv2.COLOR_BGR2RGB)
    # print(histdata)
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)


def calculatesaveHistogramGrey(number_of_bins, image_path, equal_alg):
    img_grey = cv2.imread(image_path, 0)
    img_grey_equal = equalizeHistGray(img_grey, equal_alg)
    img_hist = cv2.calcHist([img_grey_equal], [0], None, [
                            number_of_bins], [0, 256])
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, img_hist)


def normaliseHistogram(canal_hist, ):
    cv2.normalize(canal_hist, canal_hist, alpha=0, norm_type=cv2.NORM_MINMAX)


def runHistEqual(image_path, equal_alg):
    bgr_img = cv2.imread(image_path)

    ycrcb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    if equal_alg == 0:
        clahe = cv2.createCLAHE()
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    else:
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img


def equalizeHistGray(img_grey, equal_alg):
    if equal_alg == 0:
        clahe = cv2.createCLAHE()
        img_grey_equal = clahe.apply(img_grey)
    else:
        img_grey_equal = cv2.equalizeHist(img_grey)
    return img_grey_equal

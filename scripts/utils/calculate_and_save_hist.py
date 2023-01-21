import numpy as np
import cv2


def calculateHistogram(number_of_bins, image, conversion=cv2.COLOR_BGR2HSV):
    hsv_img = cv2.cvtColor(image, conversion)
    histB = cv2.calcHist([hsv_img], [0], None, [number_of_bins], [0, 256])
    histG = cv2.calcHist([hsv_img], [1], None, [number_of_bins], [0, 256])
    histR = cv2.calcHist([hsv_img], [2], None, [number_of_bins], [0, 256])

    histdata = np.row_stack((histB, histG, histR))
    return histdata


def calculateSaveHistogram(number_of_bins, image_path, normalize=0):
    img = cv2.imread(image_path)
    if normalize:
        img = normalizeHistogram(img)
    histdata = calculateHistogram(number_of_bins, img)
    # print(histdata)
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)


def calculatesaveHistogram2(number_of_bins, image_path, equal_alg):
    img = runHistEqual(image_path, equal_alg)
    # poaje konwesje bo defaukltowo jest w przestrzeni HSV
    histdata = calculateHistogram(number_of_bins, img, cv2.COLOR_BGR2RGB)
    # print(histdata)
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, histdata)


def calculateSaveHistogramGrey(number_of_bins, image_path, equal_alg=0, normalize=0):
    img_grey = cv2.imread(image_path, 0)
    if normalize:
        img_grey = normalizeHistogram(img_grey)
    if equal_alg:
        img_grey = equalizeHistGray(img_grey, equal_alg)

    img_hist = cv2.calcHist([img_grey], [0], None, [
                            number_of_bins], [0, 256])
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, img_hist)


def normalizeHistogram(img_input):
    img_normalized = cv2.normalize(img_input, 1, None, 255, cv2.NORM_MINMAX)
    return img_normalized


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

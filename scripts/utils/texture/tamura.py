import numpy as np
import cv2
from scipy.stats import moment
from scripts.utils.calculate_and_save_hist import calculateHistogram
from config import N_BINS


def saveHistTamuraFeatures(input_image_path):
    features_tam = get_tamura_features(input_image_path)
    features_hist = calculateHistogram(N_BINS, input_image_path)
    saveas = input_image_path[:input_image_path.rfind(".jpg")] + "_tam.npy"
    features = np.append(features_hist, features_tam)
    np.save(saveas, features)
    return saveas


def get_tamura_features(input_image_path) -> dict():
    """
    Function to get Tamura features of an image
Args:
        image: path to image file

Returns:
        dict: dictionary of features (coarseness, contrast, directionality)
    """
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = dict()

    features["coarseness"] = get_coarseness(gray)
    features["contrast"] = get_contrast(gray)
    features["directionality"] = get_directionality(gray)

    return list(features.values())


def get_coarseness(image):

    assert image.shape[0] > 64 and image.shape[1] >= 64, "Image dimensions should be minimum 64X64"  # noqa

    image = cv2.resize(image, (1024, 1024))
    H, W = image.shape[:2]
    Ei = []
    SBest = np.zeros((H, W))

    for k in range(1, 7):
        Ai = np.zeros((H, W))
        Ei_h = np.zeros((H, W))
        Ei_v = np.zeros((H, W))

        for h in range(2**(k-1)+1, H-(k-1)):
            for w in range(2**(k-1)+1, W-(k-1)):
                image_subset = image[h-(2**(k-1)-1): h+(2**(k-1)-1)-1,
                                     w-(2**(k-1)-1): w+(2**(k-1)-1)-1]
                Ai[h, w] = np.sum(image_subset)

        for h in range(2**(k-1)+1, H-k):
            for w in range(2 ** (k - 1) + 1, W-k):
                try:
                    Ei_h[h, w] = Ai[h+(2**(k-1)-1), w] - Ai[h-(2**(k-1)-1), w]
                    Ei_v[h, w] = Ai[h, w+(2**(k-1)-1)] - Ai[h, w-(2**(k-1)-1)]
                except IndexError:
                    pass

        Ei_h /= 2 ** (2 * k)
        Ei_v /= 2 ** (2 * k)

        Ei.append(Ei_h)
        Ei.append(Ei_v)

    Ei = np.array(Ei)
    for h in range(H):
        for w in range(W):
            maxv_index = np.argmax(Ei[:, h, w])
            k_temp = (maxv_index + 1) // 2
            SBest[h, w] = 2**k_temp

    coarseness = np.sum(SBest) / (H * W)
    return coarseness


def get_contrast(image, mask=None, n=0.25):

    H, W = image.shape[:2]
    hist = cv2.calcHist(image, [0], mask, [256], [0, 256])

    count_probs = hist / (H * W)

    std = np.std(count_probs)
    moment_4th = moment(count_probs, 4)
    kurtosis = moment_4th / (std ** 4)

    contrast = std / (kurtosis ** n)
    return contrast.item(0)


def get_directionality(image, threshold=12):
    image = np.array(image, dtype='int64')
    h = image.shape[0]
    w = image.shape[1]
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # calc for deltaH
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltaH[hi][wi] = np.sum(np.multiply(
                image[hi-1:hi+2, wi-1:wi+2], convH))
    for wi in range(w)[1:w-1]:
        deltaH[0][wi] = image[0][wi+1] - image[0][wi]
        deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

    # calc for deltaV
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltaV[hi][wi] = np.sum(np.multiply(
                image[hi-1:hi+2, wi-1:wi+2], convV))
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
    for hi in range(h)[1:h-1]:
        deltaV[hi][0] = image[hi+1][0] - image[hi][0]
        deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # calc the theta
    for hi in range(h):
        for wi in range(w):
            if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
                theta[hi][wi] = 0
            elif(deltaH[hi][wi] == 0):
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(
                    deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):  # noqa
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    return fdir

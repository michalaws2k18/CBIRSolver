import cv2
import numpy as np
from matplotlib import pyplot as plt

from scripts.utils.texture.gabor import GaborFilter
from scripts.utils.texture.tamura import get_coarseness, get_contrast, get_directionality


def getMediumBrightness(input_image_path):
    img = cv2.imread(input_image_path)
    if len(img.shape) == 2:
        return (img.flatten().mean())
    else:
        mean_B = img[:, :, 0].flatten().mean()
        mean_G = img[:, :, 1].flatten().mean()
        mean_R = img[:, :, 2].flatten().mean()
        return (mean_B, mean_G, mean_R)


def getMaximumBrightness(input_image_path):
    img = cv2.imread(input_image_path)
    if len(img.shape) == 2:
        return (int(img.flatten().max()))
    else:
        max_B = int(img[:, :, 0].flatten().max())
        max_G = int(img[:, :, 1].flatten().max())
        max_R = int(img[:, :, 2].flatten().max())
        return (max_B, max_G, max_R)


def getMinimumBrightness(input_image_path):
    img = cv2.imread(input_image_path)
    if len(img.shape) == 2:
        return (int(img.flatten().min()))
    else:
        min_B = int(img[:, :, 0].flatten().min())
        min_G = int(img[:, :, 1].flatten().min())
        min_R = int(img[:, :, 2].flatten().min())
        return (min_B, min_G, min_R)


def getStdDev(input_image_path):
    img = cv2.imread(input_image_path)
    m, s = cv2.meanStdDev(img)
    s = np.square(s)
    res = {
        'means': list(m.flatten()),
        'var': list(s.flatten())
    }
    return res


def getTamuraFeatures(input_image_path):
    img = cv2.imread(input_image_path)
    # gabor_features = GaborFilter(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = dict()

    features["coarseness"] = get_coarseness(gray)
    features["contrast"] = get_contrast(gray)
    # features["directionality"] = get_directionality(gray)

    return features


def getGaborFeatures(input_image_path):
    img = cv2.imread(input_image_path)
    gabor_features = GaborFilter(img)

    return gabor_features


def createLoadingImage(save_path, input_image_path):
    ncols = 2
    nrows = 1

    img = cv2.imread(input_image_path)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    # fig.subplots_adjust(wspace=0.2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].axis("off")
    ax[1].hist(img[:, :, 2].flatten(), bins=256, color='red')
    ax[1].hist(img[:, :, 1].flatten(), bins=256, color='green')
    ax[1].hist(img[:, :, 0].flatten(), bins=256, color='blue')

    plt.savefig(save_path)


def getInputImageValues(input_image_path):
    mean_B, mean_G, mean_R = getMediumBrightness(input_image_path)
    max_B, max_G, max_R = getMaximumBrightness(input_image_path)
    min_B, min_G, min_R = getMinimumBrightness(input_image_path)
    var = getStdDev(input_image_path)
    # tamura = getTamuraFeatures(input_image_path)
    # gabor = getGaborFeatures(input_image_path)
    image_data = {
        'image_path': input_image_path,
        'mean_blue': mean_B,
        'mean_green': mean_G,
        'mean_red': mean_R,
        'max_blue': max_B,
        'max_green': max_G,
        'max_red': max_R,
        'min_blue': min_B,
        'min_green': min_G,
        'min_red': min_R,
        'wariancja': var,
        # 'cechy Tamury': tamura,
        # 'Gabor': gabor,

    }
    return (image_data)


from scripts.metrics_quality.quality_indicators import getPrecisionAndAccuracy
from scripts.utils.texture.tamura import getTamuraFeatures
from scripts.logic.ml_model import extractFeatures
from scripts.metrics_quality.metrics_calculation import distanceManhattan, distanceEukliedian, distanceChi2
from scripts.metrics_quality.quality_indicators import getTP
from scripts.benchmarks.helper import (getTheClosestImages, createResultImage,
                                       replaceStrInListFromRight, getTheClosestImagesCoef)
from scripts.utils.calculate_and_save_hist import calculateHistogram, runHistEqual, equalizeHistGray
from config import (N_BINS, SEARCH_DIRECTORY_C, RESULT_IMAGE_PATH, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_ML,
                        SEARCH_DIRECTORY_THIN_SIFT, SEARCH_DIRECTORY_THIN_HIST_EQUAL,
                        SEARCH_DIRECTORY_THIN_HIST_EQUAL_GRAY,
                        SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE,
                        SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE_GRAY)  # noqa:E501
from scripts.utils.sift import obtainBOWVector
import cv2
import numpy as np
import os


def process_hist_solver(n_of_res, input_image_path):
    img = cv2.imread(input_image_path)
    img_hist = calculateHistogram(N_BINS, img)
    closest_images = getTheClosestImages(
        n_of_res, img_hist, SEARCH_DIRECTORY_C, distanceManhattan)
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def processHistSolverEqual(n_of_res, input_image_path, algoritm_code):
    equal_alg = algoritm_code % 2
    img = runHistEqual(input_image_path, equal_alg)
    img_hist = calculateHistogram(N_BINS, img, cv2.COLOR_BGR2RGB)
    search_folder = SEARCH_DIRECTORY_THIN_HIST_EQUAL
    if equal_alg == 0:
        search_folder = SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def processHistSolverEqualGrey(n_of_res, input_image_path, algorithm_code):
    equal_alg = algorithm_code % 2
    img_grey = cv2.imread(input_image_path, 0)
    img_grey_equal = equalizeHistGray(img_grey, equal_alg)
    img_hist = cv2.calcHist([img_grey_equal], [0], None, [N_BINS], [0, 256])
    search_folder = SEARCH_DIRECTORY_THIN_HIST_EQUAL_GRAY
    if equal_alg == 0:
        search_folder = SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE_GRAY
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def process_ml_solver(n_of_res, input_image_path):
    extracted_features = extractFeatures(input_image_path)
    closest_images = getTheClosestImages(
        n_of_res, extracted_features, SEARCH_DIRECTORY_ML, distanceManhattan)
    createResultImage(closest_images, RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def process_tamura_solver(n_of_res, input_image_path):
    img_features = getTamuraFeatures(input_image_path)
    separator = [3*N_BINS, 3*N_BINS+3]
    closest_images = getTheClosestImages(
        n_of_res, img_features, SEARCH_DIRECTORY_C_TAM, distanceManhattan, separator)
    closest_images = replaceStrInListFromRight(closest_images, '_tam', '')
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def process_hist_tamura_solver(n_of_res, input_image_path):
    tam_features = getTamuraFeatures(input_image_path)
    img_hist = calculateHistogram(N_BINS, input_image_path)
    separator = [0, 3*N_BINS, 3*N_BINS+3]
    coefficients = [0.999, 0.001]
    closest_images = getTheClosestImagesCoef(
        n_of_res, img_hist, tam_features, SEARCH_DIRECTORY_C_TAM, distanceManhattan, separator, coefficients)
    closest_images = replaceStrInListFromRight(closest_images, '_tam', '')
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def processSIFTsolver(n_of_res, input_image_path):
    sift = cv2.SIFT_create()
    centers_path = SEARCH_DIRECTORY_THIN_SIFT + os.sep + "cluster_centres.npy"
    base_centers = np.load(centers_path)
    img_bow = obtainBOWVector(input_image_path, base_centers, sift)
    closest_images = getTheClosestImages(
        n_of_res, img_bow, SEARCH_DIRECTORY_THIN_SIFT, distanceEukliedian)
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images


def processAllAlgorithms(n_of_res, input_image_path):
    res_hist = process_hist_solver(n_of_res, input_image_path)
    res_ml = process_ml_solver(n_of_res, input_image_path)
    res_hist_equal = processHistSolverEqual(n_of_res, input_image_path, 231)
    res_hist_equal_clahe = processHistSolverEqual(
        n_of_res, input_image_path, 232)
    res_hist_grey_equal = processHistSolverEqualGrey(
        n_of_res, input_image_path, 211)
    res_hist_grey_equal_clahe = processHistSolverEqualGrey(
        n_of_res, input_image_path, 212)
    precison1, recall1 = calcIndicatPrecisRecall(input_image_path, res_hist)
    TP1, FP1 = getTPandFP(input_image_path, res_hist)

    precison2, recall2 = calcIndicatPrecisRecall(input_image_path, res_ml)
    TP2, FP2 = getTPandFP(input_image_path, res_ml)

    precison3, recall3 = calcIndicatPrecisRecall(
        input_image_path, res_hist_equal)
    TP3, FP3 = getTPandFP(input_image_path, res_hist_equal)

    precison4, recall4 = calcIndicatPrecisRecall(
        input_image_path, res_hist_equal_clahe)
    TP4, FP4 = getTPandFP(input_image_path, res_hist_equal_clahe)

    precison5, recall5 = calcIndicatPrecisRecall(
        input_image_path, res_hist_grey_equal)
    TP5, FP5 = getTPandFP(input_image_path, res_hist_grey_equal)

    precison6, recall6 = calcIndicatPrecisRecall(
        input_image_path, res_hist_grey_equal_clahe)
    TP6, FP6 = getTPandFP(input_image_path, res_hist_grey_equal_clahe)

    result = {
        "histogram": {
            "closest_images": res_hist,
            "precison": precison1,
            "recall": recall1,
            "TP": TP1,
            "FP": FP1, },
        "ml": {
            "closest_images": res_ml,
            "precison": precison2,
            "recall": recall2,
            "TP": TP2,
            "FP": FP2, },
        "hist_equal": {
            "closest_images": res_hist_equal,
            "precison": precison3,
            "recall": recall3,
            "TP": TP3,
            "FP": FP3, },
        "hist_equal_clahe": {
            "closest_images": res_hist_equal_clahe,
            "precison": precison4,
            "recall": recall4,
            "TP": TP4,
            "FP": FP4, },
        "hist_grey_equal": {
            "closest_images": res_hist_grey_equal,
            "precison": precison5,
            "recall": recall5,
            "TP": TP5,
            "FP": FP5, },
        "hist_grey_equal_clahe": {
            "closest_images": res_hist_grey_equal_clahe,
            "precison": precison6,
            "recall": recall6,
            "TP": TP6,
            "FP": FP6, },
    }
    return result

# Needs some upgrade but now should work


def calcIndicatPrecisRecall(input_image_path, closest_images):
    classpath = os.path.dirname(closest_images[0][1])
    print(classpath)
    precision, recall = getPrecisionAndAccuracy(closest_images, classpath)
    print(precision)
    print(recall)
    return precision, recall

# Needs some upgrade but now should work


def getTPandFP(input_image_path, closest_images):
    classpath = os.path.dirname(closest_images[0][1])
    print(classpath)
    TP = getTP(closest_images, classpath)
    FP = len(closest_images) - TP
    return TP, FP

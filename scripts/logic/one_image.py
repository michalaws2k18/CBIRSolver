
from scripts.metrics_quality.quality_indicators import getPrecisionAndAccuracy, getPrecisionAndAccuracy2, getTPandFP
from scripts.utils.texture.tamura import getTamuraFeatures
from scripts.logic.ml_model import extractFeatures
from scripts.metrics_quality.metrics_calculation import (distanceManhattan, distanceEukliedian, distanceChi2,
                                                         distanceManhattanNorm)
from scripts.metrics_quality.quality_indicators import getTP, getQualityIndicators
from scripts.benchmarks.helper import (findAlreadyExtractedFeatures, getTheClosestImages, getTheClosestImages2, createResultImage,
                                       joinTwoDistanceLists, replaceStrInListFromRight, getTheClosestImagesCoef, getImagesInDistanceOrder)
from scripts.utils.calculate_and_save_hist import calculateHistogram, normalizeHistogram, runHistEqual, equalizeHistGray
from config import (N_BINS, SEARCH_DIRECTORY_C, RESULT_IMAGE_PATH, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_ML,
                    SEARCH_DIRECTORY_THIN_SIFT, SEARCH_DIRECTORY_HIST_EQUAL,
                    SEARCH_DIRECTORY_HIST_EQUAL_GRAY,
                    SEARCH_DIRECTORY_HIST_EQUAL_CLAHE,
                    SEARCH_DIRECTORY_HIST_EQUAL_CLAHE_GRAY,
                    SEARCH_DIRECTORY_HIST_GRAY,
                    SEARCH_DIRECTORY_HIST_GRAY_NORM,
                    SEARCH_DIRECTORY_HIST_NORM,
                    SEARCH_DIR_CCV, CCV_N, STANDALONE)
from scripts.utils.sift import obtainBOWVector
from scripts.utils.ccv import extract_CCV
from scripts.utils.decorators import timeit
import cv2
import numpy as np
import os


@timeit
def process_hist_solver(n_of_res, input_image_path, norm=0, create_image=True):
    img = cv2.imread(input_image_path)
    if norm:
        img = normalizeHistogram(img)
    img_hist = calculateHistogram(N_BINS, img)
    search_folder = SEARCH_DIRECTORY_C
    if norm:
        search_folder = SEARCH_DIRECTORY_HIST_NORM
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    if create_image:
        createResultImage(closest_images,
                          RESULT_IMAGE_PATH, n_of_res)
    return closest_images


@timeit
def processHistSolverEqual(n_of_res, input_image_path, algoritm_code, create_image=True):
    equal_alg = algoritm_code % 2
    img = runHistEqual(input_image_path, equal_alg)
    img_hist = calculateHistogram(N_BINS, img, cv2.COLOR_BGR2RGB)
    search_folder = SEARCH_DIRECTORY_HIST_EQUAL
    if equal_alg == 0:
        search_folder = SEARCH_DIRECTORY_HIST_EQUAL_CLAHE
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    if create_image:
        createResultImage(closest_images,
                          RESULT_IMAGE_PATH, n_of_res)
    return closest_images


@timeit
def processHistSolverEqualGrey(n_of_res, input_image_path, algorithm_code, create_image=True):
    equal_alg = algorithm_code % 2
    img_grey = cv2.imread(input_image_path, 0)
    img_grey_equal = equalizeHistGray(img_grey, equal_alg)
    img_hist = cv2.calcHist([img_grey_equal], [0], None, [N_BINS], [0, 256])
    search_folder = SEARCH_DIRECTORY_HIST_EQUAL_GRAY
    if equal_alg == 0:
        search_folder = SEARCH_DIRECTORY_HIST_EQUAL_CLAHE_GRAY
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    if create_image:
        createResultImage(closest_images,
                          RESULT_IMAGE_PATH, n_of_res)
    return closest_images


@timeit
def processHistSolverGreyNorm(n_of_res, input_image_path, norm=0, create_image=True):
    img_grey = cv2.imread(input_image_path, 0)
    if norm:
        img_grey = normalizeHistogram(img_grey)
    img_hist = cv2.calcHist([img_grey], [0], None, [N_BINS], [0, 256])
    search_folder = SEARCH_DIRECTORY_HIST_GRAY
    if norm != 0:
        search_folder = SEARCH_DIRECTORY_HIST_GRAY_NORM
    closest_images = getTheClosestImages(
        n_of_res, img_hist, search_folder, distanceManhattan)
    if create_image:
        createResultImage(closest_images,
                          RESULT_IMAGE_PATH, n_of_res)
    return closest_images


@timeit
def process_ml_solver(n_of_res, input_image_path, all_res=False, create_image=True):
    f_path = findAlreadyExtractedFeatures(
        input_image_path, SEARCH_DIRECTORY_ML)
    if f_path:
        extracted_features = np.load(f_path)
    else:
        extracted_features = extractFeatures(input_image_path)
    closest_images = getTheClosestImages2(
        n_of_res, extracted_features, SEARCH_DIRECTORY_ML, distanceManhattan)
    if create_image:
        createResultImage(closest_images, RESULT_IMAGE_PATH, n_of_res)
    return closest_images[:n_of_res]


@timeit
def processCCVOnlySolver(n_of_res, input_image_path, create_image=True):
    print(input_image_path)
    f_path = findAlreadyExtractedFeatures(input_image_path, SEARCH_DIR_CCV)
    if f_path:
        ccv_features = np.load(f_path)
    else:
        ccv_features = extract_CCV(input_image_path, CCV_N)
    closest_images = getTheClosestImages(
        n_of_res, ccv_features, SEARCH_DIR_CCV, distanceManhattanNorm)
    if create_image:
        createResultImage(closest_images, RESULT_IMAGE_PATH, n_of_res)
    return closest_images


@timeit
def processCCVAndHist(n_of_res, input_image_path, ccv_first=True):
    f_path = findAlreadyExtractedFeatures(input_image_path, SEARCH_DIR_CCV)
    if f_path:
        ccv_features = np.load(f_path)
    else:
        ccv_features = extract_CCV(input_image_path, CCV_N)
    img = cv2.imread(input_image_path)
    img_hist = calculateHistogram(N_BINS, img)
    search_dir_list = [SEARCH_DIR_CCV, SEARCH_DIRECTORY_C]
    features_list = [ccv_features, img_hist]
    if not ccv_first:
        search_dir_list.reverse()
        features_list.reverse()
    closest_images_0 = getImagesInDistanceOrder(
        features_list[0], search_dir_list[0], distanceManhattanNorm)
    closest_images_1 = getImagesInDistanceOrder(
        features_list[1], search_dir_list[1], distanceManhattanNorm)

    closest_images = joinTwoDistanceLists(closest_images_0, closest_images_1)
    res_closest_images = closest_images[:n_of_res]

    return res_closest_images
    # closest_images_2 = getTheClosestImages(
    #     n_of_res, features_list[1], search_dir_list[1], distanceManhattanNorm)
    # createResultImage(closest_images_final, RESULT_IMAGE_PATH, n_of_res)
    # return closest_images_final


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


def processAllAlgorithms(n_of_res, input_image_path, createImage=False, measure_time=False):
    res_hist, search_time_1 = process_hist_solver(
        n_of_res, input_image_path, create_image=createImage)
    res_hist_norm, search_time_7 = process_hist_solver(
        n_of_res, input_image_path, norm=1, create_image=createImage)
    res_hist_gray, search_time_8 = processHistSolverGreyNorm(
        n_of_res, input_image_path, create_image=createImage)
    res_hist_gray_norm, search_time_9 = processHistSolverGreyNorm(
        n_of_res, input_image_path, norm=1, create_image=createImage)
    res_ml, search_time_2 = process_ml_solver(
        n_of_res, input_image_path)
    res_hist_equal, search_time_3 = processHistSolverEqual(
        n_of_res, input_image_path, 231, create_image=createImage)
    res_hist_equal_clahe, search_time_4 = processHistSolverEqual(
        n_of_res, input_image_path, 232, create_image=createImage)
    res_hist_grey_equal, search_time_5 = processHistSolverEqualGrey(
        n_of_res, input_image_path, 211, create_image=createImage)
    res_hist_grey_equal_clahe, search_time_6 = processHistSolverEqualGrey(
        n_of_res, input_image_path, 212, create_image=createImage)
    res_ccv_only, search_time_10 = processCCVOnlySolver(
        n_of_res, input_image_path)
    res_ccv_hist, search_time_11 = processCCVAndHist(n_of_res, input_image_path,
                                                     ccv_first=True)
    # res_tamura = process_tamura_solver(n_of_res, input_image_path)

    # precison1, recall1 = calcIndicatPrecisRecall(input_image_path, res_hist)
    # TP1, FP1 = getTPandFP(input_image_path, res_hist)

    TP1, FP1, _, _, precison1, recall1 = getQualityIndicators(
        input_image_path, res_hist)

    # precison2, recall2 = calcIndicatPrecisRecall(input_image_path, res_ml)
    # TP2, FP2 = getTPandFP(input_image_path, res_ml)

    TP2, FP2, _, _, precison2, recall2 = getQualityIndicators(
        input_image_path, res_ml)

    # precison3, recall3 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_equal)
    # TP3, FP3 = getTPandFP(input_image_path, res_hist_equal)

    TP3, FP3, _, _, precison3, recall3 = getQualityIndicators(
        input_image_path, res_hist_equal)

    # precison4, recall4 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_equal_clahe)
    # TP4, FP4 = getTPandFP(input_image_path, res_hist_equal_clahe)

    TP4, FP4, _, _, precison4, recall4 = getQualityIndicators(
        input_image_path, res_hist_equal_clahe)

    # precison5, recall5 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_grey_equal)
    # TP5, FP5 = getTPandFP(input_image_path, res_hist_grey_equal)

    TP5, FP5, _, _, precison5, recall5 = getQualityIndicators(
        input_image_path, res_hist_grey_equal)

    # precison6, recall6 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_grey_equal_clahe)
    # TP6, FP6 = getTPandFP(input_image_path, res_hist_grey_equal_clahe)

    TP6, FP6, _, _, precison6, recall6 = getQualityIndicators(
        input_image_path, res_hist_grey_equal_clahe)

    # precison7, recall7 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_norm)
    # TP7, FP7 = getTPandFP(input_image_path, res_hist_norm)

    TP7, FP7, _, _, precison7, recall7 = getQualityIndicators(
        input_image_path, res_hist_norm)

    # precison8, recall8 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_gray)
    # TP8, FP8 = getTPandFP(input_image_path, res_hist_gray)

    TP8, FP8, _, _, precison8, recall8 = getQualityIndicators(
        input_image_path, res_hist_gray)

    # precison9, recall9 = calcIndicatPrecisRecall(
    #     input_image_path, res_hist_gray_norm)
    # TP9, FP9 = getTPandFP(input_image_path, res_hist_gray_norm)

    TP9, FP9, _, _, precison9, recall9 = getQualityIndicators(
        input_image_path, res_hist_gray_norm)

    # precison10, recall10 = calcIndicatPrecisRecall(
    #     input_image_path, res_ccv_only)
    # TP10, FP10 = getTPandFP(input_image_path, res_ccv_only)

    TP10, FP10, _, _, precison10, recall10 = getQualityIndicators(
        input_image_path, res_ccv_only)
    # precison11, recall11 = calcIndicatPrecisRecall(
    #     input_image_path, res_ccv_hist)
    # TP11, FP11 = getTPandFP(input_image_path, res_ccv_only)

    TP11, FP11, _, _, precison11, recall11 = getQualityIndicators(
        input_image_path, res_ccv_hist)

    # precision12, recall12 = calcIndicatPrecisRecall(
    #     input_image_path, res_tamura)
    # TP12, FP12 = getTPandFP(input_image_path, res_tamura)

    result = {
        "InceptionResNetV2": {
            "closest_images": res_ml,
            "precision": precison2,
            "recall": recall2,
            "TP": TP2,
            "FP": FP2,
            "search_time": search_time_2, },
        "histogram": {
            "closest_images": res_hist,
            "precision": precison1,
            "recall": recall1,
            "TP": TP1,
            "FP": FP1,
            "search_time": search_time_1, },
        " hist_norm": {
            "closest_images": res_hist_norm,
            "precision": precison7,
            "recall": recall7,
            "TP": TP7,
            "FP": FP7,
            "search_time": search_time_7, },
        "hist_gray": {
            "closest_images": res_hist_gray,
            "precision": precison8,
            "recall": recall8,
            "TP": TP8,
            "FP": FP8,
            "search_time": search_time_8, },
        "hist_gray_norm": {
            "closest_images": res_hist_gray_norm,
            "precision": precison9,
            "recall": recall9,
            "TP": TP9,
            "FP": FP9,
            "search_time": search_time_9, },
        "hist_gray_equal": {
            "closest_images": res_hist_grey_equal,
            "precision": precison5,
            "recall": recall5,
            "TP": TP5,
            "FP": FP5,
            "search_time": search_time_5, },
        "hist_equal": {
            "closest_images": res_hist_equal,
            "precision": precison3,
            "recall": recall3,
            "TP": TP3,
            "FP": FP3,
            "search_time": search_time_3, },
        "hist_gray_equal_clahe": {
            "closest_images": res_hist_grey_equal_clahe,
            "precision": precison6,
            "recall": recall6,
            "TP": TP6,
            "FP": FP6,
            "search_time": search_time_6, },
        "hist_equal_clahe": {
            "closest_images": res_hist_equal_clahe,
            "precision": precison4,
            "recall": recall4,
            "TP": TP4,
            "FP": FP4,
            "search_time": search_time_4, },
        "CCV": {
            "closest_images": res_ccv_only,
            "precision": precison10,
            "recall": recall10,
            "TP": TP10,
            "FP": FP10,
            "search_time": search_time_10, },
        "CCV_hist": {
            "closest_images": res_ccv_hist,
            "precision": precison11,
            "recall": recall11,
            "TP": TP11,
            "FP": FP11,
            "search_time": search_time_11, },
        # "tamura": {
        #     "closest_images": res_tamura,
        #     "precision": precision12,
        #     "recall": recall12,
        #     "TP": TP12,
        #     "FP": FP12, },
    }
    return result

# Needs some upgrade but now should work


def calcIndicatPrecisRecall(input_image_path, closest_images):
    classpath = os.path.dirname(closest_images[0][1])
    # print(classpath)
    precision, recall = getPrecisionAndAccuracy(closest_images, classpath)
    # print(precision)
    # print(recall)
    return precision, recall


# Needs some upgrade but now should work

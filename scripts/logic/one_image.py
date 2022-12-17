
from scripts.metrics_quality.quality_indicators import getPrecisionAndAccuracy
from scripts.utils.texture.tamura import getTamuraFeatures
from scripts.logic.ml_model import extractFeatures
from scripts.metrics_quality.metrics_calculation import distanceManhattan, distanceEukliedian, distanceChi2
from scripts.metrics_quality.quality_indicators import getTP
from scripts.benchmarks.helper import (getTheClosestImages, createResultImage,
                                        replaceStrInListFromRight, getTheClosestImagesCoef)
from scripts.utils.calculate_and_save_hist import calculateHistogram, runHistEqual
from config import N_BINS, SEARCH_DIRECTORY_C, RESULT_IMAGE_PATH, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_ML, SEARCH_DIRECTORY_THIN_SIFT, SEARCH_DIRECTORY_THIN_HIST_EQUAL  # noqa:E501
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

def processHistSolverEqual(n_of_res, input_image_path):
    img = runHistEqual(input_image_path)
    img_hist = calculateHistogram(N_BINS, img)
    closest_images = getTheClosestImages(
        n_of_res, img_hist, SEARCH_DIRECTORY_THIN_HIST_EQUAL, distanceManhattan)
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

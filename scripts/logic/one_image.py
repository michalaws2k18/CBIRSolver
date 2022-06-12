
from scripts.utils.texture.tamura import getTamuraFeatures
from scripts.logic.ml_model import extractFeatures
from scripts.metrics_quality.metrics_calculation import distanceManhattan
from scripts.benchmarks.helper import (getTheClosestImages, createResultImage,
                                        replaceStrInListFromRight, getTheClosestImagesCoef)
from scripts.utils.calculate_and_save_hist import calculateHistogram
from config import N_BINS, SEARCH_DIRECTORY_C, RESULT_IMAGE_PATH, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_ML  # noqa:E501


def process_hist_solver(n_of_res, input_image_path):
    img_hist = calculateHistogram(N_BINS, input_image_path)
    closest_images = getTheClosestImages(
        n_of_res, img_hist, SEARCH_DIRECTORY_C, distanceManhattan)
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

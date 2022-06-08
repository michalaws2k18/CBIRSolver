
# from utils.texture.tamura import get_tamura_features
from scripts.logic.ml_model import extractFeatures
from scripts.metrics_quality.metrics_calculation import distanceManhattan
from scripts.benchmarks.helper import getTheClosestImages, createResultImage
from scripts.utils.calculate_and_save_hist import calculateHistogram
from config import N_BINS, SEARCH_DIRECTORY_C, RESULT_IMAGE_PATH, SEARCH_DIRECTORY_ML  # noqa:E501


def process_classic_solver(n_of_res, input_image_path):
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

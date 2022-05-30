
# from utils.texture.tamura import get_tamura_features
from config import N_BINS, SEARCH_DIRECTORY
from scripts.utils.calculate_and_save_hist import calculateHistogram
from scripts.benchmarks.helper import getTheClosestImages, createResultImage
from scripts.metrics_quality.metrics_calculation import distanceManhattan
from config import RESULT_IMAGE_PATH


def process_classic_solver(n_of_res, input_image_path):
    img_hist = calculateHistogram(N_BINS, input_image_path)
    closest_images = getTheClosestImages(
        n_of_res, img_hist, SEARCH_DIRECTORY, distanceManhattan)
    createResultImage(closest_images,
                      RESULT_IMAGE_PATH, n_of_res)
    return closest_images

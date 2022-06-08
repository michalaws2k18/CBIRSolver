import os
from scripts.utils.calculate_and_save_hist import calculatesaveHistogram
from scripts.utils.texture.tamura import saveHistTamuraFeatures
from config import N_BINS, SEARCH_DIRECTORY_C, SEARCH_DIRECTORY_C_TAM


def indexImagesInDBHist(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogram(N_BINS, filepath)


def indexImagesInDBHistTam(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                saveHistTamuraFeatures(filepath)


if __name__ == '__main__':
    # indexImagesInDB(SEARCH_DIRECTORY_C)
    indexImagesInDBHistTam(SEARCH_DIRECTORY_C_TAM)

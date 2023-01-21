import os
from scripts.utils.calculate_and_save_hist import (calculateSaveHistogram, calculatesaveHistogram2,
                                                   calculateSaveHistogramGrey)
from scripts.utils.texture.tamura import saveHistTamuraFeatures
from config import (N_BINS, SEARCH_DIRECTORY_C, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_HIST_EQUAL,
                    SEARCH_DIRECTORY_HIST_EQUAL_GRAY,
                    SEARCH_DIRECTORY_HIST_EQUAL_CLAHE,
                    SEARCH_DIRECTORY_HIST_EQUAL_CLAHE_GRAY,
                    SEARCH_DIRECTORY_HIST_GRAY, SEARCH_DIRECTORY_HIST_GRAY_NORM,
                    SEARCH_DIRECTORY_HIST_NORM)


def indexImagesInDBHist(directory, norm=0):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculateSaveHistogram(N_BINS, filepath, normalize=norm)


def indexImagesInDBHistEqual(directory, equal_alg):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogram2(N_BINS, filepath, equal_alg)


def indexImagesInDBHistTam(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                saveHistTamuraFeatures(filepath)


def iterateThroughDir(index_func):
    "Decorator that iterates throufg dir"
    def wrap(dir_name, *args, **kwargs):
        for subdir, dirs, files in os.walk(dir_name):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg"):
                    index_func(filepath, *args, *kwargs)
    return wrap


@iterateThroughDir
def testwrap(filepath):
    print()


def indexImagesInDBHistEqualGrey(directory, alg, norm=0):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculateSaveHistogramGrey(
                    N_BINS, filepath, alg, normalize=norm)


if __name__ == '__main__':
    # indexImagesInDB(SEARCH_DIRECTORY_C)
    # indexImagesInDBHistTam(SEARCH_DIRECTORY_C_TAM)
    # indexImagesInDBHistEqual(SEARCH_DIRECTORY_THIN_HIST_EQUAL)
    # indexImagesInDBHistEqualGrey(
    # SEARCH_DIRECTORY_HIST_EQUAL_CLAHE_GRAY, 1)
    # indexImagesInDBHistEqual(SEARCH_DIRECTORY_HIST_EQUAL, 0)
    # indexImagesInDBHistEqualGrey(
    #     SEARCH_DIRECTORY_HIST_GRAY_NORM, alg=0, norm=1)
    indexImagesInDBHist(SEARCH_DIRECTORY_HIST_NORM, norm=1)

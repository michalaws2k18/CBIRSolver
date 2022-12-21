import os
from scripts.utils.calculate_and_save_hist import (calculatesaveHistogram, calculatesaveHistogram2,
                                                   calculatesaveHistogramGrey)
from scripts.utils.texture.tamura import saveHistTamuraFeatures
from config import (N_BINS, SEARCH_DIRECTORY_C, SEARCH_DIRECTORY_C_TAM, SEARCH_DIRECTORY_THIN_HIST_EQUAL,
                    SEARCH_DIRECTORY_THIN_HIST_EQUAL_GRAY,
                    SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE,
                    SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE_GRAY)


def indexImagesInDBHist(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogram(N_BINS, filepath)


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


def indexImagesInDBHistEqualGrey(directory, alg):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogramGrey(N_BINS, filepath, alg)


if __name__ == '__main__':
    # indexImagesInDB(SEARCH_DIRECTORY_C)
    # indexImagesInDBHistTam(SEARCH_DIRECTORY_C_TAM)
    # indexImagesInDBHistEqual(SEARCH_DIRECTORY_THIN_HIST_EQUAL)
    # indexImagesInDBHistEqualGrey(
    # SEARCH_DIRECTORY_THIN_HIST_EQUAL_GRAY, 1)
    indexImagesInDBHistEqual(SEARCH_DIRECTORY_THIN_HIST_EQUAL_CLAHE, 0)

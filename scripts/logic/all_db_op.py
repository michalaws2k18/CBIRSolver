import os
from scripts.utils.calculate_and_save_hist import calculatesaveHistogram
from config import N_BINS, SEARCH_DIRECTORY_C


def indexImagesInDB(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogram(N_BINS, filepath)


if __name__ == '__main__':
    indexImagesInDB(SEARCH_DIRECTORY_C)

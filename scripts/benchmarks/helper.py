import os
from scripts.metrics_quality.metrics_calculation import calculateDistance


def getTheClosestImages(numberofresults, inputimagehistogram, search_dir, distMetric):
    distancelist = []
    for subdir, dirs, files in os.walk(search_dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistance(
                    filepath, inputimagehistogram, distMetric)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


# def indexImagesInDB(directory, indexFunc):
#     for subdir, dirs, files in os.walk(directory):
#         for file in files:
#             filepath = subdir + os.sep + file
#             if filepath.endswith(".jpg"):
#                 calculatesaveHistogram(numberofbins, filepath)

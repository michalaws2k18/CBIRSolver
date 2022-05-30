import os
import cv2
import matplotlib.pyplot as plt
from scripts.metrics_quality.metrics_calculation import calculateDistance


def convertNpyPathsToJpeg(closest_img_paths_npy):
    imagepaths = []
    for i in range(len(closest_img_paths_npy)):
        imagepath = closest_img_paths_npy[i][1]
        imagepath = imagepath[:imagepath.rfind(".npy")] + ".jpg"
        imagepaths.append((closest_img_paths_npy[i][0], imagepath))
    return imagepaths


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
    theclosestimages_npy = distancelist[:numberofresults]
    theclosestimages = convertNpyPathsToJpeg(theclosestimages_npy)
    return theclosestimages


def createResultImage(closest_images_path, save_img_path, n_of_res):
    ncols = 4
    nrows = n_of_res//ncols
    if n_of_res % ncols != 0:
        nrows = nrows+1
    figsize = [12, 8]

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0)

    # plot simple raster image on each sub-plot
    for i in range(n_of_res):
        img = cv2.imread(closest_images_path[i][1])
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        ax[i % nrows][i // nrows].imshow(rgb_img)
        ax[i % nrows][i // nrows].axis('off')
        ax[i % nrows][i //
                      nrows].set_title("D=" + str(closest_images_path[i][0]))
        # remove_prefix(str(imagepaths[i][1]), directory)+" "
    plt.savefig(save_img_path)


# def indexImagesInDB(directory, indexFunc):
#     for subdir, dirs, files in os.walk(directory):
#         for file in files:
#             filepath = subdir + os.sep + file
#             if filepath.endswith(".jpg"):
#                 calculatesaveHistogram(numberofbins, filepath)

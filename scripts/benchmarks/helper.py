import os
import cv2
import matplotlib.pyplot as plt
from scripts.metrics_quality.metrics_calculation import calculateDistance
from config import STANDALONE
import random
from copy import deepcopy


def convertNpyPathsToJpeg(closest_img_paths_npy):
    imagepaths = []
    for i in range(len(closest_img_paths_npy)):
        imagepath = closest_img_paths_npy[i][1]
        imagepath = imagepath[:imagepath.rfind(".npy")] + ".jpg"
        imagepaths.append((closest_img_paths_npy[i][0], imagepath))
    return imagepaths


def deleteNpyfromfolder(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                os.remove(filepath)


def getTheClosestImages(
        numberofresults,
        inputimagehistogram,
        search_dir,
        distMetric,
        separator=None):
    distancelist = getListOfDistances(
        inputimagehistogram, search_dir, distMetric, separator)
    distancelist_sorted = sorted(distancelist, key=lambda tup: tup[0])
    theclosestimages_npy = distancelist_sorted[:numberofresults]
    theclosestimages_npy_norm = normalizeDistanceList(
        theclosestimages_npy, distancelist_sorted[-1][0])
    theclosestimages = convertNpyPathsToJpeg(theclosestimages_npy_norm)
    return theclosestimages


def getTheClosestImages2(
        numberofresults,
        inputimagehistogram,
        search_dir,
        distMetric,
        separator=None):
    distancelist = getListOfDistances(
        inputimagehistogram, search_dir, distMetric, separator)
    distancelist_sorted = sorted(distancelist, key=lambda tup: tup[0])
    theclosestimages_npy = deepcopy(distancelist_sorted)
    theclosestimages_npy_norm = normalizeDistanceList(
        theclosestimages_npy, distancelist_sorted[-1][0])
    theclosestimages = convertNpyPathsToJpeg(theclosestimages_npy_norm)
    return theclosestimages


def getImagesInDistanceOrder(
        inputimagehistogram,
        search_dir,
        distMetric,
        separator=None):
    distancelist = getListOfDistances(
        inputimagehistogram, search_dir, distMetric, separator)
    distancelist_sorted = sorted(distancelist, key=lambda tup: tup[0])
    theclosestimages_npy = distancelist_sorted
    theclosestimages_npy_norm = normalizeDistanceList(
        theclosestimages_npy, distancelist_sorted[-1][0])
    theclosestimages = convertNpyPathsToJpeg(theclosestimages_npy_norm)
    return theclosestimages


def getListOfDistances(input_img_hist, search_dir, dist_metric, separator=None):
    distancelist = []
    for subdir, dirs, files in os.walk(search_dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                if not filepath.endswith("cluster_centres.npy"):
                    distance = calculateDistance(
                        filepath, input_img_hist, dist_metric, separator)
                    distancelist.append((distance, filepath))
    return distancelist


def joinTwoDistanceLists(list_one, list_two):
    result_list = []
    for tuple in list_one:
        dist2 = getDistfromSecondList(list_two, tuple[1])
        result_list.append((tuple[0]+dist2, tuple[1]))
    return result_list


def matchOnlyLastTwoParts(path_one, path_two):
    last_one = os.path.split(path_one)[1]
    last_two = os.path.split(path_two)[1]
    return last_one == last_two


def getDistfromSecondList(op_list, match_filepath):
    index = next(i for i, v in enumerate(op_list)
                 if matchOnlyLastTwoParts(match_filepath, v[1]))
    return op_list[index][0]


def isSamePath(match_filepath, v):
    if match_filepath in v:
        return True
    else:
        return False


def convertPathsInList(original_list):
    result = []
    for item in original_list:
        new_path = covertPathToLocalPath(item[1])
        result.append((item[0], new_path))
    return result


def createResultImage(closest_images_path, save_img_path, n_of_res, add_hist=False):
    ncols = 4
    nrows = n_of_res//ncols
    if n_of_res % ncols != 0:
        nrows = nrows+1
    figsize = [12, 3*nrows]

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.1)

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


def createInputImageWithHist(input_image_path):
    pass


def replaceStrInListFromRight(closest_img_paths_npy, str_to_replace, replacement_str):
    imagepaths = []
    for i in range(len(closest_img_paths_npy)):
        imagepath = closest_img_paths_npy[i][1]
        imagepath = replacement_str.join(imagepath.rsplit(str_to_replace, 1))
        imagepaths.append((closest_img_paths_npy[i][0], imagepath))
    return imagepaths


def normalizeDistance(dist_value, denominator, limit=1000):
    return dist_value/denominator*limit


def normalizeDistanceList(distnace_list, denominator, limit=1000):
    normalized_list = []
    for elem in distnace_list:
        norm_dist = normalizeDistance(elem[0], denominator, limit)
        normalized_list.append((norm_dist, elem[1]))
    return normalized_list


def getTheClosestImagesCoef(
        numberofresults,
        inputimagehistogram,
        inputimageTamura,
        search_dir,
        distMetric,
        separator,
        coefficients):
    distancelistHist = getListOfDistances(
        inputimagehistogram, search_dir, distMetric, separator[:2])
    distancelistTam = getListOfDistances(
        inputimageTamura, search_dir, distMetric, separator[1:3])
    distancelistHist_sorted = sorted(distancelistHist, key=lambda tup: tup[0])
    distancelistTam_sorted = sorted(distancelistTam, key=lambda tup: tup[0])
    distancelistHist_norm = normalizeDistanceList(
        distancelistHist, distancelistHist_sorted[-1][0])
    distancelistTam_norm = normalizeDistanceList(
        distancelistTam, distancelistTam_sorted[-1][0])
    distancelistHist_norm_multi = multiplyDistanceByCoeff(
        distancelistHist_norm, coefficients[0])
    distancelistTam_norm_multi = multiplyDistanceByCoeff(
        distancelistTam_norm, coefficients[1])
    distancelist_joined = joinTwoLists(
        distancelistHist_norm_multi, distancelistTam_norm_multi)
    distancelist_joined_sorted = sorted(
        distancelist_joined, key=lambda tup: tup[0])
    distancelist_joined_limited = distancelist_joined_sorted[:numberofresults]
    distancelist_joined_limited_norm = normalizeDistanceList(distancelist_joined_limited,
                                                             distancelist_joined_sorted[-1][0])
    theclosestimages = convertNpyPathsToJpeg(distancelist_joined_limited_norm)
    return theclosestimages


def multiplyDistanceByCoeff(distance_list, coefficient):
    result = []
    for elem in distance_list:
        result.append((elem[0]*coefficient, elem[1]))
    return result


def joinTwoLists(distance_hist, distance_Tam):
    result = []
    if len(distance_hist) == len(distance_Tam):
        for index in range(len(distance_hist)):
            if distance_hist[index][1] == distance_Tam[index][1]:
                result.append(
                    (distance_hist[index][0]+distance_Tam[index][0], distance_hist[index][1]))
            else:
                print('the paths are not equal')
                result.append((None, None))
        return result
    else:
        print('distance lists not same length')
        return []


def covertPathToLocalPath(image_primary_path):
    subpath = os.path.normpath(image_primary_path)
    head, tail = os.path.split(subpath)
    head2, tail2 = os.path.split(head)
    res_path = os.path.join('http://127.0.0.1:5000/images_db/', tail2, tail)
    return res_path


def selectRandomInputImages(numberofinputsinoneclass, search_dir):
    inputimages = []
    for subdir, dirs, files in os.walk(search_dir):
        allimagesinclass = []
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                allimagesinclass.append(filepath)
        if len(allimagesinclass) > numberofinputsinoneclass:
            selectedimages = random.sample(
                allimagesinclass, numberofinputsinoneclass)
            inputimages.append(selectedimages)
    return inputimages


def findAlreadyExtractedFeatures(input_image_path, search_dir):
    if STANDALONE:
        subpath = os.path.normpath(input_image_path)
        head, tail = os.path.split(subpath)
        head2, tail2 = os.path.split(head)
        res_path = os.path.join(search_dir, tail2, tail)
        res_path = res_path[:res_path.rfind(".jpg")] + ".npy"
        return res_path
    else:
        image_name = os.path.basename(os.path.normpath(input_image_path))
        found_path = False
        for subdir, dirs, files in os.walk(search_dir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".npy"):
                    if image_name in filepath:
                        found_path = filepath
                        break

        return found_path


def convertSearchPathToLocalDir(input_image_path, search_dir):
    pass

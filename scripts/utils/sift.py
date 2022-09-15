import os
import random
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scripts.metrics_quality.metrics_calculation import distanceEukliedian
from config import EXAMPLE_IMG_PATH


def select_one_image_from_class(directory_path):
    files_list = []
    class_list = []
    for subdir, dirs, files in os.walk(directory_path):
        if len(files) > 0:
            filtered_list_iter = filter(selectOnlyJPG, files)
            filtered_list = list(filtered_list_iter)
            file = random.choice(filtered_list)
            filepath = subdir + os.sep + file
            files_list.append(filepath)
        class_list.append(subdir)

    class_list.pop(0)
    return files_list, class_list

def selectOnlyJPG(path_elem):
    if path_elem.endswith(".jpg"):
        return True
    else:
        return False

def saveList(path_name, list_to_save):
    np.save(path_name, list_to_save)

def collectDescInOne(pictures_list, model):
    all_desc = []
    for elem in pictures_list:
        img = cv2.imread(elem)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift_kp, sift_desc = model.detectAndCompute(img_grey, None)
        for desc in sift_desc:
            all_desc.append(desc)
    all_desc_np = np.array(all_desc)
    return all_desc

def obtainBOWVector(image_path, base_centers, model):
    img = cv2.imread(image_path)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift_kp, sift_desc = model.detectAndCompute(img_grey, None)
    bow_vec = np.zeros(len(base_centers))
    for desc in sift_desc:
        to_center_dist = []
        for center in base_centers:
            dist = distanceEukliedian(desc, center)
            to_center_dist.append(dist)
        min_index = to_center_dist.index(min(to_center_dist))
        bow_vec[min_index] = bow_vec[min_index] + 1
    if len(sift_desc) != bow_vec.sum():
        print('something strange when obatianing the vector')
    return bow_vec

def indexBOWVectors(DBPath, model, base_centers):
    for subdir, dirs, files in os.walk(DBPath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                features = obtainBOWVector(filepath, base_centers, model)
                saveas = filepath[:filepath.rfind(".jpg")] + ".npy"
                np.save(saveas, features)

def initialiseSIFTOps(DBPath, sample_files_list, clusters_num):
    sift = cv2.SIFT_create()
    all_descs = collectDescInOne(sample_files_list, sift)
    kmeans = KMeans(init="k-means++", n_clusters=clusters_num, n_init=20, max_iter=400, random_state=42)
    centers_path = DBPath + os.sep + "cluster_centres.npy"
    kmeans.fit(all_descs)
    saveList(centers_path, kmeans.cluster_centers_)
    return sift, kmeans

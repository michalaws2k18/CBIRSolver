from importlib.metadata import files
from scripts.utils.sift import (select_one_image_from_class, collectDescInOne, obtainBOWVector, indexBOWVectors,
                                    initialiseSIFTOps)
from sklearn.cluster import KMeans
from config import SEARCH_DIRECTORY_THIN_SIFT, EXAMPLE_IMG_PATH
from scripts.logic.one_image import processSIFTsolver
import cv2
import numpy as np

if __name__ == "__main__":
    # files_list, class_list = select_one_image_from_class(SEARCH_DIRECTORY_THIN_SIFT)
    # print(files_list)
    # print('some space \n\n\n\n')
    # sift = cv2.SIFT_create()
    # all_desc = collectDescInOne(files_list, sift)
    # print(all_desc[1])
    # number_of_classes = len(files_list)
    # kmeans = KMeans(init="k-means++", n_clusters=number_of_classes, n_init=20, max_iter=400, random_state=42)
    # kmeans.fit(all_desc)
    # print(kmeans.inertia_)
    # print('\n\n\n')
    # print(kmeans.cluster_centers_)
    # print('\n\n\n')
    # print(kmeans.n_iter_)
    # bow_vec = obtainBOWVector(EXAMPLE_IMG_PATH, kmeans.cluster_centers_, sift)
    # print(bow_vec)

    # indexing images in db process
    # print(len(files_list))
    # sift, kmeans = initialiseSIFTOps(SEARCH_DIRECTORY_THIN_SIFT, files_list, len(files_list))
    # indexBOWVectors(SEARCH_DIRECTORY_THIN_SIFT, sift, kmeans.cluster_centers_)
    # print(len(kmeans.cluster_centers_))
    # vector = np.load(r'D:\Dokumenty\CBIR\CorelDBthinSIFT\art_1\193000.npy')
    # print(len(vector))
    processSIFTsolver(10, EXAMPLE_IMG_PATH)

    print('finished :)')

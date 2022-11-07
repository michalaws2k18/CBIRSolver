from scripts.utils.sift import initialiseSIFTOps2Way, indexDBbyOwnDesc
from config import SEARCH_DIRECTORY_THIN_SIFT_2


sift, kmeans_model = initialiseSIFTOps2Way(8)
indexDBbyOwnDesc(SEARCH_DIRECTORY_THIN_SIFT_2, sift, kmeans_model)

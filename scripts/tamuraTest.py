from scripts.utils.texture.tamura import getTamuraFeatures, saveHistTamuraFeatures
from scripts.benchmarks.helper import deleteNpyfromfolder
from config import SEARCH_DIRECTORY_C_TAM
import numpy as np

if __name__ == '__main__':
    inputimagepath = r'D:\Dokumenty\CBIR\CorelDBthinTamura\pl_flower\84005.jpg'
    inputimagepathnpy = r'D:\Dokumenty\CBIR\CorelDBthinTamura\pl_flower\84005_tam.npy'
    # features = getTamuraFeatures(inputimagepath)
    # print(features)
    # all_features = saveHistTamuraFeatures(inputimagepath)
    features = np.load(inputimagepathnpy)
    print(features)
    print(features.size)

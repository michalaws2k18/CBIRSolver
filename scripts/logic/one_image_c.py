import cv2
from utils.texture.tamura import get_tamura_features
import config
from utils.calculate_and_save_hist import calculateHistogram


img = cv2.imread(config.EXAMPLE_IMG_PATH)
img_hist = calculateHistogram(config.N_BINS, config.EXAMPLE_IMG_PATH)
print(get_tamura_features(img))

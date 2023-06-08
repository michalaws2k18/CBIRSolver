from scripts.logic.one_image import calcIndicatPrecisRecall, getTPandFP, processCCVAndHist
from config import EXAMPLE_IMG_PATH
import os


n_of_res = 15
input_image_path = EXAMPLE_IMG_PATH
print('Start..')
# print(os.path.split(input_image_path))

result, _ = processCCVAndHist(n_of_res, input_image_path,
                              ccv_first=True, first_pool_multi=10)
precison11, recall11 = calcIndicatPrecisRecall(
    input_image_path, result)
TP11, FP11 = getTPandFP(input_image_path, result)
print(len(result))

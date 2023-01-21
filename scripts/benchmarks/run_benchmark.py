from scripts.logic.one_image import processAllAlgorithms
from scripts.benchmarks.helper import selectRandomInputImages
from config import SEARCH_DIRECTORY_INPUT, INPUTS_LIST
import numpy as np

number_of_results = 20
number_of_inputs_in_one_class = 5


try:
    selected_images = np.load(INPUTS_LIST)
except FileNotFoundError:
    selected_images = selectRandomInputImages(
        number_of_inputs_in_one_class, SEARCH_DIRECTORY_INPUT)
    np.save(INPUTS_LIST, selected_images)

print(len(selected_images))
print(len(selected_images[0]))


for dir_input in selected_images:
    for input_image_path in dir_input:
        pass

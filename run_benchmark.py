from scripts.logic.one_image import processAllAlgorithms
from scripts.benchmarks.helper import selectRandomInputImages
from config import SEARCH_DIRECTORY_INPUT, INPUTS_LIST, BENCHAMARK_RESULTS, BENCHAMARK_RESULTS_CSV, BASE_PATH
import numpy as np
import csv
from pprint import pprint
from copy import deepcopy
import os
import glob


def runForOneInput(n_of_res, input_image):
    all_res = processAllAlgorithms(n_of_res, input_image, measure_time=True)
    new_res = deepcopy(all_res)
    for key in all_res.keys():
        del new_res[key]['closest_images']
    row_list = []
    classpath = os.path.dirname(input_image)
    totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
    row_list.append(input_image)
    row_list.append(n_of_res)
    row_list.append(totalnumberofgoodimgindb)
    for key in new_res.keys():
        row_list.append(key)
        values = list(new_res[key].items())
        new_values = []
        for elem in values:
            new_values.extend(list(elem))
        row_list.extend(new_values)
    return row_list


if __name__ == "__main__":
    print("Hello word, here execution starts...")
    print("processing selecting input..")
    number_of_results = 20
    number_of_inputs_in_one_class = 2

    try:
        selected_images = np.load(os.path.join(BASE_PATH,INPUTS_LIST))
    except FileNotFoundError:
        selected_images = selectRandomInputImages(
            number_of_inputs_in_one_class, SEARCH_DIRECTORY_INPUT)
        np.save(INPUTS_LIST, selected_images)
    pprint(selected_images)

    res_list = []
    counter_overall = 0
    counter_class = 0
    with open(BENCHAMARK_RESULTS_CSV, 'w+') as f:
        write = csv.writer(f)
        for category in selected_images:
            print(f"In class number:{counter_class}")
            counter_class += 1
            counter_in_class = 0
            for input_image_path in category:
                print(f"Processing image {input_image_path}")
                print(f"Processing image number: {counter_overall} overall")
                print(f"Processing image number: {counter_in_class} in class")
                row_result = runForOneInput(
                    number_of_results, input_image_path)
                counter_overall += 1
                counter_in_class += 1
                res_list.append(row_result)
                write.writerow(row_result)

    np.save(BENCHAMARK_RESULTS, res_list)
    print("Sucessfully finished")

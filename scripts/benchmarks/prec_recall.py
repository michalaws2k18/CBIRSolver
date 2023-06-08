from scripts.logic.one_image import process_ml_solver, calcIndicatPrecisRecall, getTPandFP
from config import EXAMPLE_IMG_PATH, PREC_RECALL_CSV
import os
import glob
import csv

if __name__ == "__main__":
    n_of_res = 1
    classpath = os.path.dirname(EXAMPLE_IMG_PATH)
    totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
    print(
        f"Total number of images in {classpath} class is: {totalnumberofgoodimgindb}")
    closest_images, _ = process_ml_solver(
        n_of_res=10000, input_image_path=EXAMPLE_IMG_PATH)
    with open(PREC_RECALL_CSV, 'a+') as f:
        write = csv.writer(f)
        for i in range(5001, 10001):
            closest_images_cut = closest_images[:i]
            precision, recall = calcIndicatPrecisRecall(
                EXAMPLE_IMG_PATH, closest_images_cut)
            TP, FP = getTPandFP(EXAMPLE_IMG_PATH, closest_images_cut)
            write.writerow([EXAMPLE_IMG_PATH, i, precision, recall, TP, FP])

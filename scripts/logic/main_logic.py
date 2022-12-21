import logging
from scripts.benchmarks.helper import covertPathToLocalPath
from scripts.logic.one_image import (calcIndicatPrecisRecall, process_hist_solver, process_ml_solver,
                                     process_tamura_solver, process_hist_tamura_solver,
                                     processSIFTsolver, getTPandFP, processHistSolverEqual, processAllAlgorithms,
                                     processHistSolverEqualGrey)
from copy import deepcopy

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('Solver Part:')


def process_searching(search_params):
    logger.info("in main process, argumnets:")
    logger.info(search_params)
    solver_type = search_params['solver_type']
    n_of_res = search_params['n_of_res']
    input_image_path = search_params['input_image_path']
    closest_images = []
    if solver_type == 0:
        # test all solver algorithms
        res_response = processAllAlgorithms(n_of_res, input_image_path)
        logger.info('Testowanie dostępnych algorytmów')
    elif solver_type == 1:
        # ML solver used here
        logger.info('Wybrano solver ML')
        closest_images = process_ml_solver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    elif solver_type == 2:
        # Classic logic used here only histogram
        logger.info('Wybrano solver klasyczny-histogram')
        closest_images = process_hist_solver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    elif solver_type in [211, 212]:
        # histogram gray scale equalized
        logger.info(
            'Wybrano solver histogram w skali szarości zrównoważony')
        closest_images = processHistSolverEqualGrey(
            n_of_res=n_of_res,
            input_image_path=input_image_path,
            algorithm_code=int(solver_type))
        logger.info(closest_images)
    elif solver_type in [231, 232]:
        # histogram RGB equlaized
        logger.info('Wybrano solver histogram RGB zrównoważony')
        closest_images = processHistSolverEqual(
            n_of_res=n_of_res,
            input_image_path=input_image_path,
            algoritm_code=int(solver_type))
        logger.info(closest_images)
    elif solver_type == 3:
        # Classic only Tamura
        logger.info('Wybrano solver klasyczny - cechy Tamury')
        closest_images = process_tamura_solver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    elif solver_type == 4:
        # Classic histogram and Tamura features
        logger.info('Wybrano solver klasyczny - histogram oraz cechy Tamury')
        closest_images = process_hist_tamura_solver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    elif solver_type == 5:
        # Classic histogram and Tamura features
        logger.info('Wybrano solver w oparciu o deskryptor SIFT')
        closest_images = processSIFTsolver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    else:
        logger.info('Podany typ solvera nie istnieje')
    if solver_type != 0:
        return closest_images, input_image_path
    else:
        return res_response


def prepareAllData(closest_images, input_image_path):
    closest_images_local = []
    nofres = len(closest_images)
    precision, recall = calcIndicatPrecisRecall(
        input_image_path, closest_images)
    TP, FP = getTPandFP(input_image_path, closest_images)
    print(input_image_path)
    print(closest_images)
    for item in closest_images:
        item_local_path = covertPathToLocalPath(item[1])
        closest_images_local.append((item[0], item_local_path))
    # input_image_path_local = covertPathToLocalPath(input_image_path)
    print(precision)
    print(recall)
    data = {
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "FP": FP,
        "nofres": nofres,
        "closest_images_paths": closest_images_local
    }
    return data

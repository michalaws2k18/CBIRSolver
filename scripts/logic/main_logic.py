import logging
from scripts.logic.one_image_c import process_classic_solver

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('Solver Part:')


def process_searching(search_params):
    logger.info("in main process, argumnets:")
    logger.info(search_params)
    solver_type = search_params['solver_type']
    n_of_res = search_params['n_of_res']
    input_image_path = search_params['input_image_path']
    if solver_type == 1:
        # ML solver used here
        logger.info('Wybrano solver ML')
        pass
    elif solver_type == 2:
        # Classic logic used here
        logger.info('Wybrano solver klasyczny')
        closest_images = process_classic_solver(
            n_of_res=n_of_res,
            input_image_path=input_image_path)
        logger.info(closest_images)
    else:
        logger.info('Podany typ solvera nie istnieje')

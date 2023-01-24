import os
import logging

# using flask_restful
from flask import Flask, jsonify, request, send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scripts.logic.main_logic import prepareAllData, process_searching
from copy import deepcopy
from config import EXAMPLE_IMG_PATH, INPUT_IMAGE_HIST, RESULT_IMAGE_PATH
from markupsafe import escape

from scripts.utils.basic_to_all import createLoadingImage, getInputImageValues


# Variables to be defined and changed soon
UPLOAD_FOLDER = r'D:\Dokumenty\CBIR\repo_scripts\CBIRSolver\test_upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
SEARCH_IMAGE_PATH = ""  # global variable

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

# creating the flask app
app = Flask(__name__)
CORS(app)


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.

@app.route('/hello')
def about():
    return ('this works')


@app.route('/uploadImage', methods=['POST'])
def postImage():
    target = UPLOAD_FOLDER
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload")
    if 'file' not in request.files:
        logger.info('No file part')
    file = request.files['file']
    name = file.filename
    if name == '':
        logger.info('No selected file')
        name = 'sub.jpg'
    name = secure_filename(name)
    destination = "\\".join([target, name])
    logger.info("trying to save file ")
    file.save(destination)
    global SEARCH_IMAGE_PATH
    SEARCH_IMAGE_PATH = deepcopy(destination)
    createLoadingImage(INPUT_IMAGE_HIST, SEARCH_IMAGE_PATH)
    response = jsonify({'message': 'File sucessfully saved on server'})
    return response


@app.route('/getResultsImage', methods=['GET'])
def get():
    logger.info(RESULT_IMAGE_PATH)
    return send_file(RESULT_IMAGE_PATH)


@app.route('/getInputImageHist', methods=['GET'])
def getInput():
    logger.info(INPUT_IMAGE_HIST)
    return send_file(INPUT_IMAGE_HIST)


@app.route('/getInputImageValues', methods=['GET'])
def getInputValues():
    logger.info("calculating params for input image")
    exit_data = getInputImageValues(EXAMPLE_IMG_PATH)
    response = jsonify(exit_data)
    logger.info(response)
    return response


@app.route('/getResultsData', methods=['GET'])
def getResults():
    logger.info('in getting all the results data')


@app.route('/search_params', methods=['POST'])
def post():
    logger.info('Starting post search parameters')
    data = request.get_json()
    n_of_res = data['num_of_results']
    solver_type = data['solver']
    text = 'Send info about ' + str(n_of_res) + \
        ' to be send, based of ' + str(solver_type) + 'type of solver.'
    search_params = {
        'n_of_res': n_of_res,
        'solver_type': solver_type,
        'input_image_path': SEARCH_IMAGE_PATH
    }
    if solver_type != 0:
        closest_images, input_image_path = process_searching(
            search_params=search_params)
        data = prepareAllData(closest_images, input_image_path)
        exit_data = {
            'info': data,
            'image': 'http://127.0.0.1:5000/getResultsImage',
        }
    else:
        exit_data = process_searching(search_params=search_params)
    return jsonify(exit_data)


@app.route("/<path:subpath>")
def getImageByPath(subpath):
    return send_file(escape(subpath))

import os
import logging
import base64

# using flask_restful
from flask import Flask, jsonify, request, session, flash
from flask_restful import Resource, Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scripts.logic.main_logic import process_searching


# Variables to be defined and changed soon
UPLOAD_FOLDER = r'D:\Dokumenty\CBIR\repo_scripts\CBIRSolver\test_upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
search_image_path = ""

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

# creating the flask app
app = Flask(__name__)
CORS(app)
# creating an API object
api = Api(app)

# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.


class Hello(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):

        return jsonify({'message': 'hello world'})

    # Corresponds to POST request
    def post(self):

        data = request.get_json()	 # status code
        return jsonify({'data': data}), 201


# another resource to calculate the square of a number
class Square(Resource):

    def get(self, num):

        return jsonify({'square': num**2})


# my rest functions


class Image(Resource):

    def post(self):
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
        search_image_path = destination
        response = jsonify({'message': 'File sucessfully saved on server'})
        return response


class SearchParams(Resource):
    def post(self):
        logger.info('Starting post search parameters')
        data = request.get_json()
        n_of_res = data['num_of_results']
        solver_type = data['solver']
        text = 'Send info about ' + str(n_of_res) + \
            ' to be send, based of ' + str(solver_type) + 'type of solver.'
        response = jsonify({'message': text})
        search_params = {
            'n_of_res': n_of_res,
            'solver_type': solver_type,
            'input_image_path': search_image_path
        }

        process_searching(search_params=search_params)
        return response


# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Square, '/square/<int:num>')
api.add_resource(Image, '/uploadImage/')
api.add_resource(SearchParams, '/search_params/')


# driver function
if __name__ == '__main__':

    app.run(debug=True)
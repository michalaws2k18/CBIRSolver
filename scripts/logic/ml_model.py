from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os

from config import SEARCH_DIRECTORY_ML

model = InceptionResNetV2(include_top=True, weights='imagenet')
MLmodel = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)


def extractFeatures(imagePath):
    img = image.load_img(imagePath, target_size=(299, 299, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    extracted_features = model.predict_on_batch(x)
    return extracted_features


def indexDB(DBPath):
    for subdir, dirs, files in os.walk(DBPath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                features = extractFeatures(filepath)
                saveas = filepath[:filepath.rfind(".jpg")] + ".npy"
                np.save(saveas, features)


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


if __name__ == '__main__':
    indexDB(SEARCH_DIRECTORY_ML)

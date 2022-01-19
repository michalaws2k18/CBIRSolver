from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
from metrics_calculation import calculateDistanceManhattan
from quality_indicators import getPrecisionAndAccuracy
import matplotlib.pyplot as plt
from test_conf import getTheClosestImages, getListofImagesPaths, getTheClosestImagesEukliedian, getTheClosestImagesMaksimum, getTheClosestImagesManhattan, getTheClosestImagesMinkowski
import csv
from quality_indicators import getTP, getPrecisionAndAccuracyData
import glob


model = InceptionResNetV2(include_top=True, weights='imagenet')
MLmodel = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
directory = 'D:\\Dokumenty\\CBIR\\CorelDBML'
inputimagepath = r'D:\\Dokumenty\\CBIR\\CorelDBML\\pl_flower\\84010.jpg'
classpath = "D:\\Dokumenty\\CBIR\\CorelDBML\\pl_flower"
inputimagepath2 = r'D:\Dokumenty\CBIR\CorelDBML\art_1\193002.jpg'
distancelist = []
numberofresults = 12

# model2 = InceptionResNetV2(include_top=True, weights='imagenet', pooling='avg')

# img_path = r'D:\Dokumenty\CBIR\CorelDB\art_1\193000.jpg'
# img = image.load_img(img_path, target_size=(299, 299, 3))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = MLmodel.predict_on_batch(x)
# features2 = model2.predict_on_batch(x)
# print(type(features))
# print(features.size)
# print(features2.size)
# if(features == features2):
#     print("Wektory sa tozsame")
# else:
#     print("Wektory nie sa tozsame")
# file = open("features_sample.txt", "w")
# file.write(str(features))
# file.close()
# print('Predicted:', decode_predictions(features, top=3)[0])


def extractFeatures(imagePath, model: Model):
    img = image.load_img(imagePath, target_size=(299, 299, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    extracted_features = model.predict_on_batch(x)
    return extracted_features


def indexDB(DBPath, model: Model):
    for subdir, dirs, files in os.walk(DBPath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                features = extractFeatures(filepath, model)
                saveas = filepath[:filepath.rfind(".jpg")] + ".npy"
                np.save(saveas, features)


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def run_test_ML(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImages(numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend([averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(averageTP, totalnumberofgoodimgindb, totalnumberofresults)
        resultclassdata.extend([precision, recall])
        resultsdata.append(resultclassdata)
        savefilename = "resultsMLManhattan.csv"
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    return [numberofresults, avgPrecision, avgRecall]


def run_test_ML_Eukliedian(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImagesEukliedian(numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend([averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(averageTP, totalnumberofgoodimgindb, totalnumberofresults)
        resultclassdata.extend([precision, recall])
        resultsdata.append(resultclassdata)  
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    resultsdata.append([numberofresults, avgPrecision, avgRecall])
    savefilename = "MLresultsEukliedian.csv"
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)

    return [numberofresults, avgPrecision, avgRecall]


def run_test_ML_Maksimum(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImagesMaksimum(numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend([averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(averageTP, totalnumberofgoodimgindb, totalnumberofresults)
        resultclassdata.extend([precision, recall])
        resultsdata.append(resultclassdata)
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    resultsdata.append([numberofresults, avgPrecision, avgRecall])
    savefilename = "MLresultsMaksimum.csv"
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)
    return [numberofresults, avgPrecision, avgRecall]


def run_test_ML_Manhattan(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImagesManhattan(numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend([averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(averageTP, totalnumberofgoodimgindb, totalnumberofresults)
        resultclassdata.extend([precision, recall])
        resultsdata.append(resultclassdata)
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    resultsdata.append([numberofresults, avgPrecision, avgRecall])
    savefilename = "MLresultsManhattan.csv"
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)
    return [numberofresults, avgPrecision, avgRecall]


def run_test_ML_Minkowski(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImagesMinkowski(numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend([averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(averageTP, totalnumberofgoodimgindb, totalnumberofresults)
        resultclassdata.extend([precision, recall])
        resultsdata.append(resultclassdata)
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    resultsdata.append([numberofresults, avgPrecision, avgRecall])
    savefilename = "MLresultsMinkowski.csv"
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)
    return [numberofresults, avgPrecision, avgRecall]


img1 = cv2.imread(inputimagepath)
query = extractFeatures(inputimagepath, MLmodel)
# query2 = extractFeatures(inputimagepath2, MLmodel)
# dist = calculateDistance(query, query2)
# print(dist)

# indexDB(directory, MLmodel)
for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = subdir + os.sep + file
        # print(filepath)
        if filepath.endswith(".npy"):
            distance = calculateDistanceManhattan(filepath, query)
            distancelist.append((distance, filepath))

distancelist.sort(key=lambda tup: tup[0])
theclosestimages = distancelist[:numberofresults]
imagepaths = []
for i in range(len(theclosestimages)):
    imagepath = theclosestimages[i][1]
    imagepath = imagepath[:imagepath.rfind(".npy")] + ".jpg"
    imagepaths.append((theclosestimages[i][0], imagepath))


nrows, ncols = 4, 3  # array of sub-plots
figsize = [10, 8]     # figure size, inches

# prep (x,y) for extra plotting on selected sub-plots
xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
ys = np.abs(np.sin(xs))           # absolute of sine

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0)

# plot simple raster image on each sub-plot
for i in range(nrows*ncols):
    img = cv2.imread(imagepaths[i][1])
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    ax[i % nrows][i // nrows].imshow(rgb_img)
    ax[i % nrows][i // nrows].axis('off')
    ax[i % nrows][i // nrows].set_title("D=" + str(imagepaths[i][0]))

plt.show()
precison, recall = getPrecisionAndAccuracy(imagepaths, classpath)
print('Results:')
print("Precision:" + str(precison) + " Recall:" + str(recall))

import os
import math
from calculate_and_save_hist import calculatesaveHistogram, calculateHistogram
from metrics_calculation import calculateDistanceEukliedian, calculateDistanceMaksimum, calculateDistanceManhattan, calculateDistanceMinkowski
from quality_indicators import getTP, getPrecisionAndAccuracyData
import cv2
import matplotlib.pyplot as plt
import random
import glob
import csv

directory = r'D:\Dokumenty\CBIR\CorelDBML'
# inputimagepath = r'D:\Dokumenty\CBIR\CorelDB\art_1\193000.jpg'
# classpath = r'D:\Dokumenty\CBIR\CorelDB\art_1'
figsize = [10, 8]


def indexImagesInDirectory(numberofbins):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                calculatesaveHistogram(numberofbins, filepath)


def getTheClosestImages(numberofresults, inputimagehistogram):
    distancelist = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistanceManhattan(filepath, inputimagehistogram)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


def getTheClosestImagesManhattan(numberofresults, inputimagehistogram):
    distancelist = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistanceManhattan(filepath, inputimagehistogram)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


def getTheClosestImagesEukliedian(numberofresults, inputimagehistogram):
    distancelist = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistanceEukliedian(filepath, inputimagehistogram)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


def getTheClosestImagesMinkowski(numberofresults, inputimagehistogram):
    distancelist = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistanceMinkowski(filepath, inputimagehistogram)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


def getTheClosestImagesMaksimum(numberofresults, inputimagehistogram):
    distancelist = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".npy"):
                distance = calculateDistanceMaksimum(filepath, inputimagehistogram)
                distancelist.append((distance, filepath))
    distancelist.sort(key=lambda tup: tup[0])
    theclosestimages = distancelist[:numberofresults]
    return theclosestimages


def getListofImagesPaths(theclosestimages):
    imagepaths = []
    for i in range(len(theclosestimages)):
        imagepath = theclosestimages[i][1]
        imagepath = imagepath[:imagepath.rfind(".npy")] + ".jpg"
        imagepaths.append((theclosestimages[i][0], imagepath))
    return imagepaths


def showImageResults(imagepaths, numberofresults):
    images = []
    for file in imagepaths:
        image = cv2.imread(file[1])
        images.append(image)
    nrows = 5
    ncols = math.ceil(numberofresults/5)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(nrows*ncols):
        img = cv2.imread(imagepaths[i][1])
        ax[i % nrows][i // nrows].imshow(img, alpha=1)
        ax[i % nrows][i // nrows].axis('off')
        ax[i % nrows][i // nrows].set_title((str(imagepaths[i][1])).removeprefix(directory)+" "+str(imagepaths[i][0]))
    plt.show()


def selectRandomInputImages(numberofinputsinoneclass):
    inputimages = []
    for subdir, dirs, files in os.walk(directory):
        allimagesinclass = []
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                allimagesinclass.append(filepath)
        if len(allimagesinclass) > numberofinputsinoneclass:
            selectedimages = random.sample(allimagesinclass, numberofinputsinoneclass)
            inputimages.append(selectedimages)
    return inputimages


def run_test(numberofbins, numberofresults, inputpictures):
    indexImagesInDirectory(numberofbins)
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = calculateHistogram(numberofbins, inputpicture)
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
        savefilename = "resultsMinkowski_" + str(numberofbins) + "bins.csv"
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
    with open(savefilename, 'w') as f:
        write = csv.writer(f)
        write.writerows(resultsdata)
    avgPrecision = avgPrecision/len(inputpictures)
    avgRecall = avgRecall/len(inputpictures)
    return [numberofbins, avgPrecision, avgRecall]

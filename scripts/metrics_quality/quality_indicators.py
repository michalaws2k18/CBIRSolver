import glob
import os


def getTP(resultslist, classpath):
    numberofgoodcalassresults = 0
    for resultstuple in resultslist:
        if classpath in resultstuple[1]:
            numberofgoodcalassresults += 1
    return numberofgoodcalassresults


def getPrecision(resultslist, classpath):
    totalnumberofresults = len(resultslist)
    numberofgoodcalassresults = getTP(resultslist, classpath)
    precision = numberofgoodcalassresults/totalnumberofresults
    return precision


def getRecall(resultslist, classpath):
    numberofgoodcalassresults = getTP(resultslist, classpath)
    totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
    accuracy = numberofgoodcalassresults/totalnumberofgoodimgindb
    return accuracy


def getRecall2(resultslist, classpath):
    classpath = classpath.lstrip('http://127.0.0.1:5000')
    classpath = 'D:/Dokumenty/CBIR/CorelDBCleanInput'+classpath
    # print(classpath)
    numberofgoodcalassresults = getTP(resultslist, classpath)
    totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
    accuracy = numberofgoodcalassresults/totalnumberofgoodimgindb
    return accuracy


def getPrecisionAndAccuracy(resultslist, classpath):
    precison = getPrecision(resultslist, classpath)
    recall = getRecall(resultslist, classpath)
    return precison, recall


def getPrecisionAndAccuracy2(resultslist, classpath):
    precison = getPrecision(resultslist, classpath)
    recall = getRecall2(resultslist, classpath)
    return precison, recall


def getPrecisionAndAccuracyData(TPrate, tnumofgoodimgindb, tnumofres):
    precison = TPrate/tnumofres
    recall = TPrate/tnumofgoodimgindb
    return precison, recall


def getQualityIndicators(input_image_path, results_list):
    # TODO: examine_input_image_path
    classpath = os.path.dirname(results_list[0][1])
    print(input_image_path)
    print(classpath)
    TP, FP = getTPandFP(input_image_path, results_list)
    n_of_res = len(results_list)
    precision = TP/n_of_res
    one_class_size = len(glob.glob1(classpath, "*.jpg"))
    recall = TP/one_class_size
    return TP, FP, n_of_res, one_class_size, precision, recall


def getANMRR():
    pass


def getTPandFP(input_image_path, closest_images):
    # TODO: examine_input_image_path
    classpath = os.path.dirname(closest_images[0][1])
    # print(classpath)
    TP = getTP(closest_images, classpath)
    FP = len(closest_images) - TP
    return TP, FP

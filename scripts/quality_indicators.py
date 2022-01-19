import glob


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


def getPrecisionAndAccuracy(resultslist, classpath):
    precison = getPrecision(resultslist, classpath)
    recall = getRecall(resultslist, classpath)
    return precison, recall


def getPrecisionAndAccuracyData(TPrate, tnumofgoodimgindb, tnumofres):
    precison = TPrate/tnumofres
    recall = TPrate/tnumofgoodimgindb
    return precison, recall

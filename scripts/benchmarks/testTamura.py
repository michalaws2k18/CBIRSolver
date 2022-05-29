

def run_test_ML(numberofresults, inputpictures):
    resultsdata = []
    avgPrecision = 0
    avgRecall = 0
    for inputpicturesclass in inputpictures:
        resultclassdata = []
        for inputpicture in inputpicturesclass:
            inputimagehistogram1 = extractFeatures(inputpicture, MLmodel)
            theclosestimages = getTheClosestImages(
                numberofresults, inputimagehistogram1)
            theclosestimagepaths = getListofImagesPaths(theclosestimages)
            # showImageResults(theclosestimagepaths, numberofresults)
            classpath = os.path.dirname(inputpicture)
            TPrate = getTP(theclosestimagepaths, classpath)
            resultclassdata.append(TPrate)
        totalnumberofresults = len(theclosestimagepaths)
        totalnumberofgoodimgindb = len(glob.glob1(classpath, "*.jpg"))
        averageTP = sum(resultclassdata)/len(resultclassdata)
        resultclassdata.insert(0, classpath)
        resultclassdata.extend(
            [averageTP, totalnumberofresults, totalnumberofgoodimgindb])
        precision, recall = getPrecisionAndAccuracyData(
            averageTP, totalnumberofgoodimgindb, totalnumberofresults)
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

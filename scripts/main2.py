from test_conf import selectRandomInputImages, run_test
import cv2
import numpy as np
import csv


numberofinputinoneclass = 8



inputpicslist = selectRandomInputImages(numberofinputinoneclass)


# inputimagepath = r'D:\Dokumenty\CBIR\CorelDB\art_1\193000.jpg'
# im = cv2.imread(inputimagepath)
# print(type(im))
# print(im.dtype)

binsnumb = [2**j for j in range(2, 10)]
print(binsnumb)

allshortQualInd = []
for binnumber in binsnumb:
    shortQualInd = run_test(binnumber, 20, inputpicslist)
    allshortQualInd.append(shortQualInd)
print(allshortQualInd)
with open('finalresultMinkowski.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(allshortQualInd)

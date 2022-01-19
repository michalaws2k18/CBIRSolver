import os
from calculate_and_save_hist import calculatesaveHistogram, calculateHistogram
from metrics_calculation import calculateDistanceManhattan
from quality_indicators import getPrecisionAndAccuracy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time 


directory = 'D:\Dokumenty\CBIR\CorelDB'
binsnumber = 32
inputimagepath = r'D:\Dokumenty\CBIR\CorelDB\pl_flower\84010.jpg'
classpath = r'D:\Dokumenty\CBIR\CorelDB\pl_flower'
inputimagepath2 = r'D:\Dokumenty\CBIR\CorelDB\art_1\193002.jpg'
distancelist = []
numberofresults = 12


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


img1 = cv2.imread(inputimagepath)
# cv2.imshow('image',img1)
# for subdir, dirs, files in os.walk(directory):
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith(".jpg"):
#             calculatesaveHistogram(binsnumber, filepath)

inputimagehistogram1 = calculateHistogram(binsnumber, inputimagepath)

# totalnumberofgoodimgindb = len(glob.glob1(inputimagepath, "*.jpg"))
# print(totalnumberofgoodimgindb)
# print(glob.glob1(classpath, "*.jpg"))

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = subdir + os.sep + file
        # print(filepath)
        if filepath.endswith(".npy"):
            distance = calculateDistanceManhattan(filepath, inputimagehistogram1)
            distancelist.append((distance, filepath))

distancelist.sort(key=lambda tup: tup[0])
theclosestimages = distancelist[:numberofresults]
imagepaths = []
for i in range(len(theclosestimages)):
    imagepath = theclosestimages[i][1]
    imagepath = imagepath[:imagepath.rfind(".npy")] + ".jpg"
    imagepaths.append((theclosestimages[i][0], imagepath))

print(imagepaths)
# images = []
# for file in imagepaths:
#     image = cv2.imread(file[1])
#     images.append(image)


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
    # remove_prefix(str(imagepaths[i][1]), directory)+" "

plt.show()
precison, recall = getPrecisionAndAccuracy(imagepaths, classpath)
print('Results:')
print("Precision:" + str(precison) + " Recall:" + str(recall))

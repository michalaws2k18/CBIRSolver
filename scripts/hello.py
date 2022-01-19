import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

img = cv2.imread('lena.png')
px = img[100, 100]
print(px)
print(img.shape[0] * img.shape[1])
# accessing only blue pixel
blue = img[100, 100, 0]
print(blue)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
bins = [[]]
for i in range(256):
    if(i % 16 == 0):
        bins.append(i)
bins.append(255)
# print(bins)
color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [16], [0, 256])
#     histdata.append(histr)
#     plt.plot(histr, color=col)
# plt.show()

histB = cv2.calcHist([img], [0], None, [16], [0, 256])
histG = cv2.calcHist([img], [1], None, [16], [0, 256])
histR = cv2.calcHist([img], [2], None, [16], [0, 256])


histdata = np.column_stack((histB, histG, histR))


# print(histdata)


directory = 'D:\Dokumenty\CBIR\CorelDB'

for subdir, dirs, files in os.walk(directory):
    # print(dirs)
    # for file in files:
    #     #print os.path.join(subdir, file)
    #     filepath = subdir + os.sep + file

    #     if filepath.endswith(".jpg"):
    #         print(filepath)
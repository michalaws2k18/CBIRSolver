import os
import random
from config import (SEARCH_DIRECTORY_THIN_SIFT, SEARCH_DIRECTORY_THIN_SIFT_2, SEARCH_DIRECTORY_THIN_HIST_EQUAL,
                    SEARCH_DIRECTORY_THIN_HIST_NORM)


directory = SEARCH_DIRECTORY_THIN_HIST_NORM

# for subdir, dirs, files in os.walk(directory):
#     fileslist = []
#     flag1 = False
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith(".jpg"):
#             fileslist.append(filepath)
#         flag1 = True
#     if flag1:
#         toremove = random.sample(fileslist, len(fileslist)-40)
#         for element in toremove:
#             os.remove(element)

# for subdir, dirs, files in os.walk(directory):
#     print(len(dirs))

for subdir, dirs, files in os.walk(directory):
    fileslist = []
    flag1 = False
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".npy"):
            fileslist.append(filepath)
        flag1 = True
    if flag1:
        for element in fileslist:
            os.remove(element)

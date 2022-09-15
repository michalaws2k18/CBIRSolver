import cv2
import numpy as np
from config import EXAMPLE_IMG_PATH
from matplotlib import pyplot as plt

img = cv2.imread(EXAMPLE_IMG_PATH)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create SURF Feature Detector object
fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()
sift = cv2.SIFT_create()

kp = fast.detect(img, None)
orb_kp, orb_desc = orb.detectAndCompute(img, None)
sift_kp, sift_desc = sift.detectAndCompute(img_grey, None)
print(len(sift_kp))
print(len(sift_desc))
print(len(sift_desc[0]))

print(sift_desc[0])
print(orb_kp[1])

print(kp[0])
img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
cv2.imwrite('fast_true.png', img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv2.imwrite('fast_false.png', img3)

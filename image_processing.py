# Image Processing with OpenCV

import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# verification des versions de python et openCV
print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)

print("OPENCV Version =", cv2.__version__)

rep_cour = os.getcwd()
print(rep_cour)

# importation de l'image a analyser et conversion couleurs vers niveaux de gris
img = cv2.imread('Images2020/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# seuillage binaire
ret_binaire, img_binaire = cv2.threshold(img, 80, 90, cv2.THRESH_BINARY)

# seuillage adaptatif
# img_adaptive_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
img_adaptive_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("seuillage binaire THRESH_BINARY", img_adaptive_thresholding)

"""
jeu1 = cv2.imread('Images2020/jeu1.jpg')
jeu1 = cv2.cvtColor(jeu1, cv2.COLOR_BGR2GRAY)

jeu2 = cv2.imread('Images2020/jeu2.jpg')
jeu2 = cv2.cvtColor(jeu2, cv2.COLOR_BGR2GRAY)

jeu3 = cv2.imread('Images2020/jeu3.jpg')
jeu3 = cv2.cvtColor(jeu3, cv2.COLOR_BGR2GRAY)

difference1 = cv2.subtract(jeu2, jeu1)
difference2 = cv2.subtract(jeu1, jeu2)
difference = difference1 + difference2
ret_binaire, difference = cv2.threshold(difference, 80, 200, cv2.THRESH_BINARY_INV)

cv2.imshow("difference", difference)
"""

"""
image = cv2.imread('Images2020/cellules.png')
img_binaire = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_close = cv2.morphologyEx(img_binaire, cv2.MORPH_CLOSE, np.ones(5))
image_erode = cv2.erode(img_binaire, np.ones(10), 20)

CCWSbase = cv2.connectedComponentsWithStats(img_binaire, 8, cv2.CV_32S)
CCWSmodified = cv2.connectedComponentsWithStats(image_erode, 8, cv2.CV_32S)

print("nb labels base : " + str(CCWSbase[0]))
print("nb labels modified : " + str(CCWSmodified[0]))

cv2.imshow("image-binaire", img_binaire)
cv2.imshow("image_erode", image_erode)
cv2.imshow("image_close", image_close)
"""

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

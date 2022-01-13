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


# 2D convolution (averaging filter) (partie 2 question 1)
kernel_5 = np.ones((5, 5), np.float32) / 25
dst5 = cv2.filter2D(img, -1, kernel_5)
cv2.imshow("kernel=5 blur", dst5)

kernel_9 = np.ones((9,9),np.float32)/(9**2)
dst9 = cv2.filter2D(img, -1, kernel_9)
cv2.imshow("kernel=9 blur", dst9)

kernel_15 = np.ones((15,15),np.float32)/(15**2)
dst15 = cv2.filter2D(img, -1, kernel_15)
cv2.imshow("kernel=15 blur", dst15)

# Gaussian blurring + median blurring (partie 2 question 2)
"""
gaussianBlur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("gaussian blur (5,5)", gaussianBlur)

medianBlur = cv2.medianBlur(img,5)
cv2.imshow("median blur (5)", medianBlur)
"""

# Partie 2 question 3 : application des filtres de sobel
"""
img_zebre = cv2.imread('zebre.jpg')
img_zebre = cv2.cvtColor(img_zebre, cv2.COLOR_BGR2GRAY)
sobelx_zebre = cv2.Sobel(img_zebre, cv2.CV_64F, 1, 0, ksize=5)
cv2.imshow("filtre sobel vertical (5)", sobelx_zebre)

img_suzan = cv2.imread('suzan.jpg')
img_suzan = cv2.cvtColor(img_suzan, cv2.COLOR_BGR2GRAY)
sobely_suzan = cv2.Sobel(img_suzan, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow("filtre sobel horizontal (5)", sobely_suzan)
"""
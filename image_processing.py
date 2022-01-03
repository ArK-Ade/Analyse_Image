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

"""
# calcule le laplacien de img
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# application du filtre de sobel sur img, verticalement (sobelx) et horizontalement (sobely)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)


# affichage des differentes Images2020 calculees
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
"""

"""
fig = plt.figure("Egalisation de l'histogramme")
# egalisation de l'histogramme de niveau de gris et comparaison avec celui non egalise
equalized_img = cv2.equalizeHist(img)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(equalized_img, cmap='gray')
plt.title('equalized'), plt.xticks([]), plt.yticks([])

# histogramme de niveaux de gris de img et de sa version egalisee
plt.subplot(2, 2, 3), plt.hist(img.ravel(), 256, [0, 256])
plt.title('original histogram') #, plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.hist(equalized_img.ravel(), 256, [0, 256])
plt.title('equalized histogram') #, plt.xticks([]), plt.yticks([])
"""

# 2D convolution (averaging filter) (partie 2 question 1)
"""
kernel_5 = np.ones((5, 5), np.float32) / 25
dst5 = cv2.filter2D(img, -1, kernel_5)
cv2.imshow("kernel=5 blur", dst5)

kernel_9 = np.ones((9,9),np.float32)/(9**2)
dst9 = cv2.filter2D(img, -1, kernel_9)
cv2.imshow("kernel=9 blur", dst9)

kernel_15 = np.ones((15,15),np.float32)/(15**2)
dst15 = cv2.filter2D(img, -1, kernel_15)
cv2.imshow("kernel=15 blur", dst15)
"""

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

# Partie 3 question 1 :
"""
# seuillage binaire
ret_binaire, img_binaire = cv2.threshold(img, 80, 90, cv2.THRESH_BINARY)

# seuillage adaptatif
#img_adaptive_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
img_adaptive_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("seuillage binaire THRESH_BINARY", img_adaptive_thresholding)
"""
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
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

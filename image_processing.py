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

# importation de l'image à analyser et conversion couleurs vers niveaux de gris
img = cv2.imread('Images2020/lisa.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calcule le laplacien de img
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# application du filtre de sobel sur img, verticalement (sobelx) et horizontalement (sobely)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# affichage des differentes images calculees
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

fig = plt.figure()

# egalisation de l'histogramme de niveau de gris et comparaison avec celui non egalise
equalized_img = cv2.equalizeHist(img)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(equalized_img, cmap='gray')
plt.title('equalized'), plt.xticks([]), plt.yticks([])

# histogramme de niveaux de gris de img et de sa version egalisee
plt.subplot(2, 2, 3), plt.hist(img.ravel(), 256, [0, 256])
plt.title('original histogram')  # , plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.hist(equalized_img.ravel(), 256, [0, 256])
plt.title('equalized histogram')  # , plt.xticks([]), plt.yticks([])

# image en niveau de gris et ses versions filtrées (passe bas)

img = cv2.imread('Images2020/lisa.png')


plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(blur1), plt.title('Blurred 5*5')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(blur2), plt.title('Blurred 9*9')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(blur3), plt.title('Blurred 15*15')
plt.xticks([]), plt.yticks([])
plt.show()

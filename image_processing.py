# Image Processing with OpenCV

# Importation des librairies
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

# Affichage le chemin du code source du programme
rep_cour = os.getcwd()
print(rep_cour)

# importation de l'image à analyser et conversion couleurs vers niveaux de gris
img = cv2.imread('Images2020/lisa.png', 0)

# application du filtre laplacien et du filtre de sobel sur img, verticalement (sobelx) et horizontalement (sobely)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
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
plt.show()

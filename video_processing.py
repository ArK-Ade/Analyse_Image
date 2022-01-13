# Mon script OpenCV : Video_processing

# importations des librairies necessaires au programme
import numpy as np
import cv2


# fonction de traitement d'une frame
def frame_processing(imgc):
    # application de la détection de contours par l'algorithme de canny
    edges = cv2.Canny(imgc, 100, 200)

    laplace = cv2.Laplacian(imgc, cv2.CV_64F)

    sobel = cv2.Sobel(imgc, cv2.CV_64F, 0, 1, ksize=5)

    return edges


# important du fichier video a traiter
cap = cv2.VideoCapture('Images2020/jurassicworld.mp4')

# le programme boucle tant que la video n'est pas finie (ou bien touche 'q' pressee)
while (True):

    # recuperation de la prochaine frame de la video
    ret, frame = cap.read()

    # si une frame a bien ete recuperee...
    if ret == True:

        try:
            img_previous = img

        except:
            img_previous = frame.copy()

        # on en fait une copie et on la convertit en niveaux de gris
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # application de la fonction de traitement frame_processing() a gray
        gray = frame_processing(gray)

        # affichage de la frame avant et apres traitement
        cv2.imshow('MavideoAvant', frame)
        cv2.imshow('MavideoApres', gray)

    else:
        print('video ended')
        break

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

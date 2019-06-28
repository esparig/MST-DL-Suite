import os
import numpy as np
import cv2

def inspect_images():

    imgH = cv2.imread("D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI\ObjImg_20180713_102730.863_271_7_H.png", 2)
    print(imgH.shape)
    for i in range(imgH.shape[0]):
        for j in range(imgH.shape[1]):
            print(imgH[i][j], end=" ")
        print()

    cv2.imshow('image',imgH)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgS = cv2.imread("D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI\ObjImg_20180713_102730.863_271_7_S.png", 2)
    print(imgS.shape)
    for i in range(imgS.shape[0]):
        for j in range(imgS.shape[1]):
            print(imgS[i][j], end=" ")
        print()

    cv2.imshow('image',imgS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgI = cv2.imread("D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI\ObjImg_20180713_102730.863_271_7_I.png", 2)
    print(imgI.shape)
    for i in range(imgI.shape[0]):
        for j in range(imgI.shape[1]):
            print(imgI[i][j], end=" ")
        print()

    cv2.imshow('image',imgI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgB1 = cv2.imread("D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI\ObjImg_20180713_102730.863_271_7_AuxB1.png", 2)
    max = 0
    print(imgB1.shape)
    for i in range(imgB1.shape[0]):
        for j in range(imgB1.shape[1]):
            print(imgB1[i][j], end=" ")
            if imgB1[i][j] > max:
                max = imgB1[i][j]
        print()
    print("Max value for AuxB1: ", max)
    cv2.imshow('image',imgB1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgB2 = cv2.imread("D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI\ObjImg_20180713_102730.863_271_7_AuxB2.png", 2)
    max = 0
    print(imgB2.shape)
    for i in range(imgB2.shape[0]):
        for j in range(imgB2.shape[1]):
            print(imgB2[i][j], end=" ")
            if imgB2[i][j] > max:
                max = imgB2[i][j]
        print()
    print("Max value for AuxB2: ", max)

    cv2.imshow('image',imgB2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #data = np.array([cv2.imread(name) for name in os.listdir('D:\OneDrive\OneDrive - MULTISCAN TECHNOLOGIES S.L\muestras\pruebaAceitunasHSI')], dtype=np.float64)

inspect_images()

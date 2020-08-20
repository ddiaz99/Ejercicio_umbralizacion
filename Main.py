import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    ruta_imagen = input('Ingrese la ruta de la imagen: ')
    imagen_original = cv2.imread(ruta_imagen,1)
    I_YCrCb = cv2.cvtColor(imagen_original,cv2.COLOR_BGR2YCrCb)
    ret, Ibw_otsu = cv2.threshold(I_YCrCb[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Ibw_Cb_mask = cv2.bitwise_not(Ibw_otsu)
    I_HSV = cv2.cvtColor(imagen_original,cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([I_HSV], [0], Ibw_Cb_mask, [180], [0, 180])
    max_pos = int(hist_hsv.argmax())

    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    Ibw_H_mask = cv2.inRange(I_HSV, lim_inf, lim_sup)
    ret, Ibw_S_mask = cv2.threshold(I_HSV[:,:,1], 128, 255, cv2.THRESH_BINARY)
    Ibw_mask = cv2.bitwise_and(Ibw_H_mask,Ibw_S_mask)
    cv2.imshow('Imagen',Ibw_mask)
    cv2.waitKey(0)
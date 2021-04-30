'''
Created on 29. apr. 2021

@author: Bruger
'''
import cv2
import numpy as np
def openClosePowerTrans(img):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    c=1; gamma = 0.5;
    powerRet = c*(double(closing))^gamma;
    return powerRet
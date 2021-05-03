'''
Created on 29. apr. 2021

@author: Bruger
'''
import cv2
import numpy as np
def openClose(img):
    #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8) #3x3 kernel
    kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8) #5x5 kernel
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    return np.expand_dims(closing, 2)

def PowerTrans(img):
    c=1; gamma = 0.5;
    powerRet = c*(img)**gamma;
    return powerRet

def median(img):
    medianImg = cv2.medianBlur(img,3)
    return np.expand_dims(medianImg,2)

def CLAHE(img):
    temp = 255.*img#To avoid messing with the reference img
    imgUint = temp.astype(np.uint8) #Convert to uint 8 to make CLAHe work.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(imgUint)
    
    cl1 = cl1.astype(np.float32)
    cl1 /=255.
    
    return np.expand_dims(cl1,2)
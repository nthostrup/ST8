'''
Created on 25. mar. 2021

@author: Bruger
'''


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Conv2D, Conv2DTranspose, concatenate
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
import pydicom
import PIL
from PIL import ImageOps

#Import own modules
import MRI_generator_module


#Makes a generator to feed network with tuple (input,masks)
def make_generator(input_img_paths, mask_paths, batch_size, img_size):
    
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(mask_paths)
       
    generator = MRI_generator_module.MRI_generator(batch_size, img_size, input_img_paths, mask_paths)
    
    
    return generator

#Load MRI images
def input_loader(path):
    #TODO: Fix sort to be numerical and not binary
    input_images = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
            ]
        )
        
    return input_images


#Load MASKS
def mask_loader(path):
    #TODO: Fix sort to be numerical and not binary
    masks = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".png")
            ]
        )
    
        
    return masks

#Helper to plot image and mask pair
def plot_img_mask(img_path,mask_path):
    
    #TODO: Change if input is jpg
    file_to_open=pydicom.dcmread(img_path)
    
    plt.figure()
    plt.imshow(file_to_open.pixel_array,cmap='gray')
    
    
    #TODO: Figure out if loading is with plt.imread or keras.image.load_img
    
    #maskImg=load_img(mask_path, color_mode="grayscale")
    #maskImg=np.expand_dims(maskImg, 2)
    
    maskImg=plt.imread(mask_path)
    plt.figure()
    plt.imshow(maskImg,cmap='gray')
    
    plt.show()









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
import utility_module as utils
import model_module

#Global variables/hyperparameters
INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/T2_imgs/2A01_P1_D1/"
MASK_DIR = "C:/Users/Bruger/Desktop/MRI/T2_masks/mask/"
BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = (512,512)
NUM_CHANNELS_OUT = 3

#Load image and mask paths
input_img_paths = utils.input_loader(INPUT_DIR)
mask_img_paths = utils.mask_loader(MASK_DIR)


#Print paths
for input_path, target_path in zip(input_img_paths[:], mask_img_paths[:]):
    print(input_path, "|", target_path)

train_gen = utils.make_generator(input_img_paths, mask_img_paths, BATCH_SIZE, IMG_SIZE)#Randomizes lists inlist

print("\n \n")
for input_path, target_path in zip(input_img_paths[:5], mask_img_paths[:5]): #Check if lists randomized
    print(input_path, "|", target_path)

#Plot of mask and DICOM image
#utils.plot_img_mask(input_img_paths[2], mask_img_paths[2])

#Get model, input dimensions must match generator. Last dimension added since it must be present
model = model_module.get_model((512,512,1), NUM_CHANNELS_OUT)

#Train model and get history of training performance
history, model = model_module.train_model(model, train_gen, train_gen, EPOCHS)

#Load model from directory
#model = load_model("Unet_MRI_1-2.h5")#Load

#Validate model and plot image, mask and prediction
predictions = model_module.test_model(model, train_gen, input_img_paths, mask_img_paths)

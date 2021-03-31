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
import PIL
from PIL import ImageOps

#Import own modules
import utility_module as utils
import model_module

#Global variables/hyperparameters
TRAIN_INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/Training/Img"
TRAIN_MASK_DIR = "C:/Users/Bruger/Desktop/MRI/Training/Mask"
VALID_INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/Validation/Img"
VALID_MASK_DIR ="C:/Users/Bruger/Desktop/MRI/Validation/Mask"
TEST_INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/Test/Img"
TEST_MASK_DIR ="C:/Users/Bruger/Desktop/MRI/Test/Mask"
BATCH_SIZE = 20
EPOCHS = 20
IMG_SIZE = (512,512)
NUM_CHANNELS_OUT = 1

#Load image and mask paths - training
training_img_paths = utils.input_loader(TRAIN_INPUT_DIR)
training_mask_paths = utils.mask_loader(TRAIN_MASK_DIR)

#Load image and mask paths - validation
validation_img_paths = utils.input_loader(VALID_INPUT_DIR)
validation_img_paths = validation_img_paths[:140]#Shorten training time
validation_mask_paths = utils.mask_loader(VALID_MASK_DIR)
validation_mask_paths = validation_mask_paths[:140]#Shorten training time

#Load image and mask paths - Test
test_img_paths = utils.input_loader(TEST_INPUT_DIR)
test_img_paths = test_img_paths[:140]#Shorten training time
test_mask_paths = utils.mask_loader(TEST_MASK_DIR)
test_mask_paths = test_mask_paths[:140]#Shorten training time

#Print paths
for input_path, target_path in zip(training_img_paths[:20], training_mask_paths[:20]):
    print(input_path, "|", target_path)

train_gen = utils.make_generator(training_img_paths, training_mask_paths, BATCH_SIZE, IMG_SIZE)#Randomizes lists inlist
valid_gen = utils.make_generator(validation_img_paths, validation_mask_paths, BATCH_SIZE, IMG_SIZE)#Randomizes lists inlist
test_gen = utils.make_generator(test_img_paths, test_mask_paths, BATCH_SIZE, IMG_SIZE)#Randomizes lists inlist

print("\n \n")
for input_path, target_path in zip(training_img_paths[:5], training_mask_paths[:5]): #Check if lists randomized
    print(input_path, "|", target_path)

#Plot of mask and image
utils.plot_img_mask(validation_img_paths[2], validation_mask_paths[2])

#Get model, input dimensions must match generator. Last dimension added since it must be present
model = model_module.get_model(IMG_SIZE, NUM_CHANNELS_OUT)

#Train model and get history of training performance
history, model = model_module.train_model(model, valid_gen, test_gen, EPOCHS) #TODO: FEJLER I VALIDATION STEP

#Load model from directory
#model = load_model("Unet_MRI_1-2.h5")#Load

#Validate model and plot image, mask and prediction
predictions = model_module.test_model(model, test_gen, test_img_paths, test_mask_paths)

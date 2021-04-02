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
                if fname.endswith(".jpeg")
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
                if fname.endswith(".jpeg")
            ]
        )
    
        
    return masks

#Helper to plot image and mask pair
def plot_img_mask(img_path,mask_path):
    
    #TODO: Change if input is jpg
    img = load_img(img_path)
    img_arr = img_to_array(img)
    img_arr /= 255.
    
    plt.figure()
    plt.imshow(img_arr,cmap='gray')
    
    
    #TODO: Figure out if loading is with plt.imread or keras.image.load_img
    
    #maskImg=load_img(mask_path, color_mode="grayscale")
    #maskImg=np.expand_dims(maskImg, 2)
    
    maskImg=plt.imread(mask_path)
    plt.figure()
    plt.imshow(maskImg,cmap='gray')
    
    plt.show()

def plot_training_history(history):
    
    #Plot performance from training
    #TODO: Move to utility function
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    #val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    #plt.plot(epochs, acc, 'bo', label='Training acc')
    #plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    #plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show(block=False)


def plot_predictions(predictions, input_img_paths, mask_paths):
    #Selected slice to compare
    i = 6;
    #Plots 
    #plt.subplot(1,2,1)
    predicted_mask = predictions[i]
    rounded = np.round(predicted_mask,0)
    plt.figure()
    plt.imshow(rounded,cmap='gray')
    
    #plt.subplot(1,2,2)
    #Print paths to ensure that they match
    print("input path: " + input_img_paths[i])
    print("mask path: " + mask_paths[i])
    
    #Plot MRI and mask via utility module
    plot_img_mask(input_img_paths[i], mask_paths[i])
    
    #Plot mask alone
    #maskImg=plt.imread(mask_paths[i])
    #plt.imshow(maskImg,cmap='gray')
    #plt.show()
    



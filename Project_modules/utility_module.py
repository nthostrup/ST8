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
from tensorflow.keras import backend as K
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
    
    img = load_img(img_path)
    img_arr = img_to_array(img)
    img_arr /= 255.
    
    plt.figure()
    plt.imshow(img_arr,cmap='gray')
    
       
    #maskImg=load_img(mask_path, color_mode="grayscale")
    #maskImg=np.expand_dims(maskImg, 2)
    
    maskImg=plt.imread(mask_path)
    plt.figure()
    plt.imshow(maskImg,cmap='gray')
    
    plt.show()

def plot_training_history(history):
    
    #Plot performance from training
    #TODO: Consider plotting dice measure
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    #plt.plot(epochs, acc, 'bo', label='Training acc')
    #plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.legend()
    
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show(block=False)


def plot_predictions(predictions, input_img_paths, mask_paths):
    #Selected slice to compare
    i = 7;
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
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
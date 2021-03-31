'''
Created on 26. mar. 2021

@author: Bruger
'''
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#Import own modules
import utility_module as utils

#Make model 
def get_model(img_size, num_classes):
    #Input shape, must match the generated images in MRI_Generator class (variable x)
    inputs = keras.Input(shape=img_size+(1,))
    
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    #Encoder
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    #Decoder
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)#sigmoid og num_classes =1
    #Mathias: Antallet af klasser skal v�re 1, da vi skal have mange pixels med 1 eller 0, og activation skal dermed v�re sigmoid og loss binary.
    
    model = keras.Model(inputs, outputs)
    
    return model

#Train model
def train_model(model, train_generator, validation_generator, epochs):
    #Compile model dependant on output dimensions
    #TODO: Implement metric DICE
    #Metric MeanIoU:  IOU is defined as follows: IOU = true_positive / (true_positive + false_positive + false_negative
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics = [keras.metrics.MeanIoU(num_classes=2)])#Works with 2 classes as output from model.
    model.summary()
    
    #TODO: Implement callbacks
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-checkpoint_Unet_MRI.h5', verbose=1, save_best_only=True)
    callback = [earlystopper, checkpointer]

    
    # Train the model, doing validation at the end of each epoch.
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callback)#Training with callbacks
    #history = model.fit(train_generator, epochs=epochs)#Training with callbacks
    
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
    plt.show()
    
    #Save model
    model.save('Unet_MRI-1.h5')
    
    return history, model

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator, input_img_paths, mask_paths):
    #TODO: Maybe calculate dice based on prediction and ground truth
        
    #TODO: Evaluate with some sample images (x_test) and ground truth masks (y_test)
    #result = model.evaluate(x_test,y_test, batch_size=32)
    
    
    predictions = model.predict(test_generator)
    
    #Selected slice to compare
    i = 6;
    #Plots 
    #plt.subplot(1,2,1)
    plt.imshow(predictions[i],cmap='gray')
    
    #plt.subplot(1,2,2)
    #Print paths to ensure that they match
    print("input path: " + input_img_paths[i])
    print("mask path: " + mask_paths[i])
    
    #Plot MRI and mask via utility module
    utils.plot_img_mask(input_img_paths[i], mask_paths[i])
    
    #Plot mask alone
    #maskImg=plt.imread(mask_paths[i])
    #plt.imshow(maskImg,cmap='gray')
    #plt.show()
    
    
    
    
    return predictions
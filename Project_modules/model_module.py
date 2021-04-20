'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
import utility_module as utils
import numpy as np
import math

#Import own modules


#Make model 
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(1,))

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1) # dims (256, 256, 16)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2) # dims (None, 128, 128, 32)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  #dims (None, 64, 64, 64)  

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)


    u5 = layers.Conv2D(64, (3, 3), padding='same')(c4)
    u5 = UpSampling2D((2,2))(u5) # dims (None, 128, 128, 64)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2D(32, (3, 3), padding='same')(c5)
    u6 = UpSampling2D((2,2))(u6) # dims (None, 256, 256, 32)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2D(16, (3, 3), padding='same')(c6)
    u7 = UpSampling2D((2,2))(u7) # dims (None, 512, 512, 16)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    print(outputs)

    return model

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator):
    #TODO: Maybe calculate dice based on prediction and ground truth
        
    #TODO: Evaluate with some sample images (x_test) and ground truth masks (y_test)
    #result = model.evaluate(x_test,y_test, batch_size=32)
    loss, f1_score, precision, recall = model.evaluate(test_generator, verbose=1)
    
    batch_size = test_generator.batch_size
    all_samples = len(test_generator.mask_paths)
    num_batches = math.floor(all_samples/batch_size)  # floor forces it to round down
    
    predictions = model.predict(test_generator)#, steps=n_batch) 
    total_dice_pixelw = utils.total_dice_pixelwise(predictions, num_batches, batch_size, test_generator) 
    print("Total pixelwise Dice is: ")
    print(total_dice_pixelw)
        
    return predictions, loss, f1_score, precision, recall


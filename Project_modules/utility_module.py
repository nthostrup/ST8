'''
Created on 25. mar. 2021

@author: Bruger
'''


import os
import random
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
    i = 5;
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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#K.clip klipper hvis y_true*y_pred ligger udenfor arrayet [0,1]
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) #K.round afrunder ved 0.5, K.sum tager summen
    recall = true_positives / (possible_positives + K.epsilon()) #K.epsilon er en meget lille vaerdi så vi ikke dividerer med 0
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true,y_pred)
    recall = recall_m(y_true,y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_imagewise_m(y_true, y_pred):
    f1_scores = []
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    for k in range(len(y_true)):
        # Dette er tjekket for om der er en tom maske med en tom prediction - det virker ikke lige nu for model.fit
        nonzero_labels = np.count_nonzero(y_true[k])
        rounded = np.round(y_pred[k], 0)   #afrunder =>0.50 = 1      <0.50 = 0
        nonzero_predictions = np.count_nonzero(rounded)
        if nonzero_labels == 0 and nonzero_predictions <= 1:
            #print("ZERO MASK", k)
            f1_scores.append(np.nan)
        else:
            #print(k)

            precision = precision_m(y_true[k], y_pred[k])
            recall = recall_m(y_true[k], y_pred[k])

            f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
            f1_scores.append(f1_score)
    mean_dice = np.nanmean(f1_scores)
    return mean_dice

def dice_imagewise_m_test(y_true, y_pred):
    f1_scores = []
    for k in range(len(y_true)):
        nonzero_labels = np.count_nonzero(y_true[k])
        rounded = np.round(y_pred[k], 0)   #afrunder =>0.50 = 1      <0.50 = 0
        nonzero_predictions = np.count_nonzero(rounded)
        if nonzero_labels == 0 and nonzero_predictions <= 1:
            #print("ZERO MASK", k)
            f1_scores.append(np.nan)
        else:
            #print(k)

            precision = precision_m(y_true[k], y_pred[k])
            recall = recall_m(y_true[k], y_pred[k])

            f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
            f1_scores.append(f1_score)

    mean_dice = np.nanmean(f1_scores)
    #print("Imagewise dice pr batch:", mean_dice)
    return mean_dice
  
def dice_pixelwise_variables(y_true, y_pred):
   
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP = np.multiply(TP, 1.0) #typecast to float
    
    truemask_negative=(K.round(K.clip(y_true, 0, 1))==0) # 1 when truemask is black 
    predmask_positive=(K.round(K.clip(y_pred, 0, 1))==1) # 1 when predmask is white 
    truemask_positive=(K.round(K.clip(y_true, 0, 1))==1) # 1 when truemask is white
    predmask_negative=(K.round(K.clip(y_pred, 0, 1))==0) # 1 when predmask is black 
    
    FP_temp=(truemask_negative & predmask_positive)
    FP_temp=np.multiply(FP_temp, 1.0) #convert from false, true.. to 0, 1... 
    FP=K.sum(FP_temp)
    
    FN_temp = (truemask_positive & predmask_negative)
    FN_temp = np.multiply(FN_temp, 1.0)
    FN=K.sum(FN_temp) 

    return TP,FP,FN

def dice_pixelwise(TP,FP,FN):
    dice=(2*TP)/((TP+FP)+(TP+FN))
    return dice
    
def total_dice_pixelwise(predictions, num_batches, batch_size, test_generator):
    #Calculate 
    #n_batch=1;
    TPsum=0 #initially TP, FP or FN is set to zero 
    FPsum=0
    FNsum=0
     
    for n_batch in range(1, num_batches+1):
        predictions_batch =predictions[(n_batch-1)*batch_size:n_batch*batch_size]
        #determine ground truth
        inputs, mask = test_generator.__getitem__(n_batch-1) #getitem input is batchindex. 
        true_mask = mask.astype('float32') 
    
        TP, FP, FN = dice_pixelwise_variables(true_mask, predictions_batch)
        TPsum = TPsum+TP
        FPsum = FPsum+FP
        FNsum = FNsum+FN
        #calculate variables for pixelwisedice 
        #print("TP:",TP,"FP:",FP,"FN:",FN)
        #dice_pixelw=dice_pixelwise(TP, FP, FN)
        #print("Batch pixelwise Dice is: ")
        #print(dice_pixelw)
        
    total_dice_pixelw=dice_pixelwise(TPsum, FPsum, FNsum) 
    return total_dice_pixelw
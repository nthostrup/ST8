'''
Created on 25. mar. 2021

@author: Bruger
'''


import os
import random
import numpy as np
from tensorflow.keras import backend as K

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
    masks = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".jpeg")
            ]
        )
    
        
    return masks



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) #K.clip klipper hvis y_true*y_pred ligger udenfor arrayet [0,1]
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   #K.round afrunder ved 0.5, K.sum tager summen
    recall = true_positives / (possible_positives + K.epsilon()) #K.epsilon er en meget lille værdi så vi ikke dividerer med 0
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

def dice_imagewise(y_true, y_pred):
    f1_scores = []
    for k in range(len(y_true)):
        nonzero_labels = np.count_nonzero(y_true[k])
        rounded = np.round(y_pred[k], 0)   #afrunder =>0.50 = 1      <0.50 = 0
        nonzero_predictions = np.count_nonzero(rounded)
        if nonzero_labels == 0 and nonzero_predictions <= 1:
            #print("ZERO MASK", k)
            f1_scores.append(np.nan)
        else:
                           
            TP, FP, FN = dice_pixelwise_variables(y_true[k], y_pred[k]) #Get TP, FP, FN for the given image/mask
            f1_score = dice_pixelwise(TP, FP, FN) #Calculate the dice for the given image/mask
            
            '''
            precision = precision_m(y_true[k], y_pred[k])
            recall = recall_m(y_true[k], y_pred[k])

            f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
            #f1_score=f1_m(y_true[k], y_pred[k])
            '''
            f1_scores.append(f1_score)

    mean_dice = np.nanmean(f1_scores)
    #print("Imagewise dice pr batch:", mean_dice)
    return mean_dice

def dice_pixelwise_variables(y_true, y_pred):
   
    TP = np.sum(np.around(np.clip(y_true * y_pred, 0, 1)))
    TP = np.multiply(TP, 1.0) #typecast to float
    
    truemask_negative=(np.around(np.clip(y_true, 0, 1))==0) # 1 when truemask is black 
    predmask_positive=(np.around(np.clip(y_pred, 0, 1))==1) # 1 when predmask is white 
    truemask_positive=(np.around(np.clip(y_true, 0, 1))==1) # 1 when truemask is white
    predmask_negative=(np.around(np.clip(y_pred, 0, 1))==0) # 1 when predmask is black 
    
    FP_temp=(truemask_negative & predmask_positive)
    FP_temp=np.multiply(FP_temp, 1.0) #convert from false, true.. to 0, 1... 
    FP=np.sum(FP_temp)
    
    FN_temp = (truemask_positive & predmask_negative)
    FN_temp = np.multiply(FN_temp, 1.0)
    FN=np.sum(FN_temp) 

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
        _, mask = test_generator.__getitem__(n_batch-1) #getitem input is batchindex. 
        true_mask = mask.astype('float32') 
    
        TP, FP, FN = dice_pixelwise_variables(true_mask, predictions_batch)
        TPsum = TPsum+TP
        FPsum = FPsum+FP
        FNsum = FNsum+FN
        
    total_dice_pixelw=dice_pixelwise(TPsum, FPsum, FNsum) 
    return total_dice_pixelw


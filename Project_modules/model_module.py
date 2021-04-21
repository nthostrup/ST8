'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
import utility_module as utils
import numpy as np
import math

#Import own modules
import MRI_generator_module


#Make model 
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(1,))

    #downsampling/encoder
    ec1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    ec1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(ec1)
    
    p1 = layers.MaxPooling2D((2, 2))(ec1) # dims (256, 256, 16)

    ec2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    ec2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(ec2)
    
    p2 = layers.MaxPooling2D((2, 2))(ec2) # dims (None, 128, 128, 32)

    ec3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    ec3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(ec3)
    
    p3 = layers.MaxPooling2D((2, 2))(ec3)  #dims (None, 64, 64, 64)  

    ec4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    ec4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(ec4)
    
    
    
    ##Upsampling/decoder
    #convolution and upsampling block
    u3 = layers.Conv2D(128, (3, 3), padding='same')(ec4)
    u3 = UpSampling2D((2,2))(u3) # dims (None, 128, 128, 64)
    
    u3 = layers.concatenate([u3, ec3])
    
    dc3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    dc3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(dc3)

    #convolution and upsampling block
    u2 = layers.Conv2D(64, (3, 3), padding='same')(dc3)
    u2 = UpSampling2D((2,2))(u2) # dims (None, 256, 256, 32)
    
    u2 = layers.concatenate([u2, ec2])
    
    dc2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    dc2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(dc2)

    #convolution and upsampling block
    u1 = layers.Conv2D(32, (3, 3), padding='same')(dc2)
    u1 = UpSampling2D((2,2))(u1) # dims (None, 512, 512, 16)
    
    u1 = layers.concatenate([u1, ec1])
    
    dc1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    dc1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dc1)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(dc1)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator):
    batch_size = test_generator.batch_size
    all_samples = len(test_generator.mask_paths)
    num_batches = math.floor(all_samples/batch_size)  # floor forces it to round down
    
    predictions = model.predict(test_generator)#, steps=n_batch) 
    
    #IMAGEWISE DICE
    f1_pr_batch = []
    for i in range(num_batches):
        _, mask = test_generator.__getitem__(i)
        mask = mask.astype('float32')  # typecast from uint8 to float32
        # faa kun 1 batch ind ad gangen
        prediction_batch = predictions[i*batch_size:(i+1)*batch_size]
        
        f1_scores = utils.dice_imagewise(y_true=mask, y_pred=prediction_batch)  # beregn f1 score for single prediction
        
        f1_pr_batch.append(f1_scores)       #Saves each
    
    mean_dice_imagewise = np.mean(f1_pr_batch)
   
    # PIXELWISE DICE
    
    total_dice_pixelwise = utils.total_dice_pixelwise(predictions, num_batches, batch_size, test_generator) 

    return predictions, mean_dice_imagewise, total_dice_pixelwise

    f1_mean = np.mean(f1_pr_batch)
    print("F1_mean", f1_mean)

    return predictions, f1_mean


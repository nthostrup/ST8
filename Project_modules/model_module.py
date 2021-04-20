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


#Import own modules


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
    u1 = layers.Conv2D(128, (3, 3), padding='same')(ec4)
    u1 = UpSampling2D((2,2))(u1) # dims (None, 128, 128, 64)
    
    u1 = layers.concatenate([u1, ec3])
    
    dc1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    dc1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(dc1)

    #convolution and upsampling block
    u2 = layers.Conv2D(64, (3, 3), padding='same')(dc1)
    u2 = UpSampling2D((2,2))(u2) # dims (None, 256, 256, 32)
    
    u2 = layers.concatenate([u2, ec2])
    
    dc2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    dc2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(dc2)

    #convolution and upsampling block
    u3 = layers.Conv2D(32, (3, 3), padding='same')(dc2)
    u3 = UpSampling2D((2,2))(u3) # dims (None, 512, 512, 16)
    
    u3 = layers.concatenate([u3, ec1])
    
    dc3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u3)
    dc3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dc3)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(dc3)

    model = Model(inputs=[inputs], outputs=[outputs])
    print(outputs)

    return model

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator):
    #TODO: Maybe calculate dice based on prediction and ground truth
        
    #TODO: Evaluate with some sample images (x_test) and ground truth masks (y_test)
    #result = model.evaluate(x_test,y_test, batch_size=32)
    loss, f1_score, precision, recall = model.evaluate(test_generator, verbose=1)

    predictions = model.predict(test_generator, steps=1)

    
    
    return predictions, f1_score


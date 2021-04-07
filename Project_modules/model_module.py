'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#Import own modules


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
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
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
    #Mathias: Antallet af klasser skal være 1, da vi skal have mange pixels med 1 eller 0, og activation skal dermed være sigmoid og loss binary.
    
    model = keras.Model(inputs, outputs)
    
    return model


#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator):
    #TODO: Maybe calculate dice based on prediction and ground truth
        
    #TODO: Evaluate with some sample images (x_test) and ground truth masks (y_test)
    #result = model.evaluate(x_test,y_test, batch_size=32)
    
    
    predictions = model.predict(test_generator, steps = 1)
    
    
    return predictions


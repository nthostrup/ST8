'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.core import Dropout, Lambda


#Import own modules


#Make model 
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(1,))

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    print(outputs)

    return model


#Train model
def train_model(model, train_generator, validation_generator, epochs):
    #Compile model dependant on output dimensions
    #TODO: Implement metric DICE
    #Metric MeanIoU:  IOU is defined as follows: IOU = true_positive / (true_positive + false_positive + false_negative
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics = [keras.metrics.MeanIoU(num_classes=2)])#Works with 2 classes as output from model.
    model.summary()
    
    earlystopper = EarlyStopping(monitor='val_loss',patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-checkpoint_Unet_MRI.h5', monitor='val_loss', verbose=1, save_best_only=True)
    callback = [earlystopper, checkpointer]

    # Train the model, doing validation at the end of each epoch.
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callback)#Training with validation as generator or dataset
    #history = model.fit(train_generator, epochs=epochs, validation_data=validation_data, validation_batch_size=val_batch_size, callbacks=callback)#Training with validation as tuple
    
    
    #history = model.fit(train_generator, epochs=epochs, callbacks=callback)#Training without validation
    
    
    #Save model
    #model.save('Unet_MRI-2.h5')
    
    return history, model

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model,test_generator):
    #TODO: Maybe calculate dice based on prediction and ground truth
        
    #TODO: Evaluate with some sample images (x_test) and ground truth masks (y_test)
    #result = model.evaluate(x_test,y_test, batch_size=32)
    
    
    predictions = model.predict(test_generator, steps = 1)
    
    
    
    
    
    return predictions


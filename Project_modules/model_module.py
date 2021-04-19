'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
import utility_module as utils
import numpy as np
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Lambda


#Import own modules
import MRI_generator_module


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

#Test model with test data(or other data) and plot prediction vs. image vs. true mask
def test_model(model, test_generator, validation_mask_paths):
    """Calculates the metrics on single images, and afterwards take the mean of all images"""
    # kræver på nuværende tidspunkt at batchsize = 1 - det kan man evt. definere inden denne metode kaldes.

    # prøv at tjekke om den kører pr. sample eller pr. batch. Evt. lave en løkke så vi kun får en dice pr. billede og derefter mean
    # Undersøge pixel wise dice eller gennemsnit pr. billede

    predictions = []
    f1_scores = []
    recalls = []
    precisions = []
    for k in range(validation_mask_paths.__len__()*test_generator.batch_size):
        print(k)
        inputs, mask = test_generator.__getitem__(k)
        mask = mask.astype('float32')   # typecast fra uint8 til float32
        #print("Inputs:", inputs)
        #print("X axis size of inputs:", np.size(inputs, 0))        # Input dimensioner i x aksen (rækker) Batchsizex512x512
        #print("Y axis size of inputs:", np.size(inputs, 1))        # Input dimensioner i y aksen (kolonner)
        #print("Z axis size of inputs:", np.size(inputs, 2))        # Input dimensioner i y aksen (kolonner)
        prediction = model.predict(inputs)  #Lav single prediction på item k
        predictions.append(prediction)  #Save in array
        #print("Predictions:", prediction)
        #print("X axis size of predictions:", np.size(prediction, 0))
        #print("Y axis size of predictions:", np.size(prediction, 1))
        #print("Z axis size of predictions:", np.size(prediction, 2))
        f1_test = utils.f1_m(y_true=mask, y_pred=prediction)    #beregn f1 score for single prediction

        recall = utils.recall_m(mask, prediction)
        precision = utils.precision_m(mask, prediction)
        if f1_test is not None:
            f1_scores.append(f1_test)   #Save in array
            recalls.append(recall)
            precisions.append(precision)
            #print("F1 scores:", f1_test)
            #print("Len of F1scores:", len(f1_scores))

    print(f1_scores[0])
    print(f1_scores[1])
    print(f1_scores[2])
    print(f1_scores[3])
    print(f1_scores[4])
    print(f1_scores[5])
    print(f1_scores[6])
    print(f1_scores[7])
    print(f1_scores[8])
    print(f1_scores[9])
    print(f1_scores[10])

    mean_f1 = np.mean(f1_scores)
    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    print("Mean f1", mean_f1)
    print("Mean recall", mean_recall)
    print("Mean precision", mean_precision)

    return predictions, mean_f1, mean_recall, mean_precision


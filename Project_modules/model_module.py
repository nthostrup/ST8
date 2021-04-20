'''
Created on 26. mar. 2021

@author: Bruger
'''


from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
import utility_module as utils
import math
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
def test_model(model, test_generator):
    """Calculates the metrics on single images, and afterwards take the mean of all images"""
    #Få batches ind i funktionerne

    batch_size = test_generator.batch_size
    all_samples = len(test_generator.mask_paths)
    num_batches = math.floor(all_samples/batch_size)  # floor forces it to round down

    predictions = []
    f1_pr_batch = []
    counter = 0
    for i in range(num_batches):
        counter = counter + 1
        print(counter)
        inputs, mask = test_generator.__getitem__(i)
        mask = mask.astype('float32')  # typecast from uint8 to float32

    # få kun 1 batch ind ad gangen
        predictions = model.predict(inputs)  # Smid et helt batch ind
        f1_scores = utils.f1_m_test(y_true=mask, y_pred=predictions)  # beregn f1 score for single prediction
        f1_pr_batch.append(f1_scores)       #Saves each

    f1_mean = np.mean(f1_pr_batch)
    print("F1_mean", f1_mean)

    return predictions, f1_mean


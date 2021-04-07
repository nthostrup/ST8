'''
Created on 31. mar. 2021

@author: Bruger
'''
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Import own modules
import utility_module as utils
import model_module

class main_class:
    def __init__(self, train_input_dir,train_mask_dir,valid_input_dir,valid_mask_dir,test_input_dir,test_mask_dir):
        self.TRAIN_INPUT_DIR = train_input_dir
        self.TRAIN_MASK_DIR = train_mask_dir
        self.VALID_INPUT_DIR = valid_input_dir
        self.VALID_MASK_DIR = valid_mask_dir
        self.TEST_INPUT_DIR = test_input_dir
        self.TEST_MASK_DIR = test_mask_dir
        self.BATCH_SIZE = 20
        self.EPOCHS = 2
        self.IMG_SIZE = (512,512)
        self.NUM_CHANNELS_OUT = 1
        
        self.SAMPLES_TO_RUN = 100
    def run_main(self):
        #Load image and mask paths - training
        training_img_paths = utils.input_loader(self.TRAIN_INPUT_DIR)
        training_img_paths = training_img_paths[:self.SAMPLES_TO_RUN]
        training_mask_paths = utils.mask_loader(self.TRAIN_MASK_DIR)
        training_mask_paths = training_mask_paths[:self.SAMPLES_TO_RUN]
        
        #Load image and mask paths - validation
        validation_img_paths = utils.input_loader(self.VALID_INPUT_DIR)
        validation_img_paths = validation_img_paths[:self.SAMPLES_TO_RUN]#Shorten training time
        validation_mask_paths = utils.mask_loader(self.VALID_MASK_DIR)
        validation_mask_paths = validation_mask_paths[:self.SAMPLES_TO_RUN]#Shorten training time
        
        #Load image and mask paths - Test
        test_img_paths = utils.input_loader(self.TEST_INPUT_DIR)
        test_mask_paths = utils.mask_loader(self.TEST_MASK_DIR)
        
        
        #Print paths
        #for input_path, target_path in zip(training_img_paths[:20], training_mask_paths[:20]):
        #    print(input_path, "|", target_path)
        
        train_gen = utils.make_generator(training_img_paths, training_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
        
        valid_gen = utils.make_generator(validation_img_paths, validation_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
               
        test_gen = utils.make_generator(test_img_paths, test_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
        
        
        
        #print("\n \n")
        #for input_path, target_path in zip(training_img_paths[:5], training_mask_paths[:5]): #Check if lists randomized
        #    print(input_path, "|", target_path)
        
        #Plot of mask and image
        #utils.plot_img_mask(training_img_paths[2], training_mask_paths[2])
        
        #Get model, input dimensions must match generator. Last dimension added since it must be present
        model = model_module.get_model(self.IMG_SIZE, self.NUM_CHANNELS_OUT)
        
        #Train model and get history of training performance
        history, model = self.train_model(model, train_gen, valid_gen, self.EPOCHS)
               
        #Plot history of training
        utils.plot_training_history(history)
        
        #Load model from directory
        #model = load_model("Unet_MRI-1.h5")#Load
        
        #Validate model and plot image, mask and prediction
        predictions = model_module.test_model(model, valid_gen)
        
        #Plot predictions
        utils.plot_predictions(predictions, validation_img_paths, validation_mask_paths)
        
    def train_model(self,model, train_generator, validation_generator, epochs):
        #Compile model dependant on output dimensions
        #TODO: Implement metric DICE
        #Metric MeanIoU:  IOU is defined as follows: IOU = true_positive / (true_positive + false_positive + false_negative
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics = [keras.metrics.MeanIoU(num_classes=2)])#Works with 2 classes as output from model.
        model.summary()
        
        
        earlystopper = EarlyStopping(monitor='val_loss',patience=5, verbose=1)
        checkpointer = ModelCheckpoint('model-checkpoint_Unet_MRI.h5', verbose=1, monitor='val_loss', save_best_only=True)
        callback = [earlystopper, checkpointer]
    
        
        # Train the model, doing validation at the end of each epoch.
        history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callback)#Training with validation as generator or dataset
               
        #Save model
        model.save('Unet_MRI-2.h5')
        
        return history, model
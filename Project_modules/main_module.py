'''
Created on 31. mar. 2021

@author: Bruger
'''
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#Import own modules
import utility_module as utils
import model_module
from tensorflow.python.distribute.coordinator.cluster_coordinator import InputError
#import plotting_module

output_data = '/output_data'

class main_class:
    def __init__(self, train_input_dir, train_mask_dir ,train_slice_dir, valid_input_dir, valid_mask_dir, valid_slice_dir, test_input_dir, test_mask_dir, test_slice_dir):
        self.TRAIN_INPUT_DIR = train_input_dir
        self.TRAIN_MASK_DIR = train_mask_dir
        self.TRAIN_SLICE_DIR = train_slice_dir
        
        self.VALID_INPUT_DIR = valid_input_dir
        self.VALID_MASK_DIR = valid_mask_dir
        self.VALID_SLICE_DIR = valid_slice_dir
        
        self.TEST_INPUT_DIR = test_input_dir
        self.TEST_MASK_DIR = test_mask_dir
        self.TEST_SLICE_DIR = test_slice_dir
        
        self.BATCH_SIZE = 50
        self.EPOCHS = 100
        self.IMG_SIZE = (512, 512)
        self.NUM_CHANNELS_OUT = 1

        self.SAMPLES_TO_RUN = -1
    def run_main(self):
        #Load image and mask paths - training
        training_img_paths = utils.input_loader(self.TRAIN_INPUT_DIR)
        training_img_paths = training_img_paths[:self.SAMPLES_TO_RUN]
        
        training_mask_paths = utils.mask_loader(self.TRAIN_MASK_DIR)
        training_mask_paths = training_mask_paths[:self.SAMPLES_TO_RUN]
        
        training_slice_paths = utils.input_loader(self.TRAIN_SLICE_DIR)
        training_slice_paths = training_slice_paths[:self.SAMPLES_TO_RUN]
        

        #Load image and mask paths - validation
        validation_img_paths = utils.input_loader(self.VALID_INPUT_DIR)
        validation_img_paths = validation_img_paths[:self.SAMPLES_TO_RUN]#Shorten training time
        
        validation_mask_paths = utils.mask_loader(self.VALID_MASK_DIR)
        validation_mask_paths = validation_mask_paths[:self.SAMPLES_TO_RUN]#Shorten training time
        
        validation_slice_paths = utils.input_loader(self.VALID_SLICE_DIR)
        validation_slice_paths = validation_slice_paths[:self.SAMPLES_TO_RUN]#Shorten training time
        

        #Load image and mask paths - Test
        test_img_paths = utils.input_loader(self.TEST_INPUT_DIR)
        
        test_mask_paths = utils.mask_loader(self.TEST_MASK_DIR)
        
        test_slice_paths = utils.input_loader(self.TEST_SLICE_DIR)


#         #Print paths
#         for input_path, target_path, slice_path in zip(training_img_paths[:20], training_mask_paths[:20], training_slice_paths[:20]):
#             print(input_path, "|", target_path, "/", slice_path)
        
        if(nrInputChannels == 1):
            train_gen = utils.make_generator(training_img_paths, training_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#With 1 channel generator. Randomizes lists inlist
            valid_gen = utils.make_generator(validation_img_paths, validation_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#With 1 channel generator. Randomizes lists inlist
            test_gen = utils.make_generator(test_img_paths, test_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#with 1 channel generator.Randomizes lists inlist
        elif(nrInputChannels == 2):
            train_gen = utils.make_generator_w_slicenumber(training_img_paths, training_mask_paths, self.BATCH_SIZE, self.IMG_SIZE,training_slice_paths)#With 2 channel generator. Randomizes lists inlist
            valid_gen = utils.make_generator_w_slicenumber(validation_img_paths, validation_mask_paths, self.BATCH_SIZE, self.IMG_SIZE,validation_slice_paths)#with 2 channel generator. Randomizes lists inlist
            test_gen = utils.make_generator_w_slicenumber(test_img_paths, test_mask_paths, self.BATCH_SIZE, self.IMG_SIZE,test_slice_paths)#with 2 channel generator.Randomizes lists inlist
        else:
            print("Input nr. of input channels!")
            raise InputError

#         print("\n \n")
#         for input_path, target_path, slice_path in zip(training_img_paths[-20:], training_mask_paths[-20:],training_slice_paths[-20:]): #Check if lists randomized
#             print(input_path, "|", target_path, "/", slice_path)
        
        
        #Get model, input dimensions must match generator. Last dimension added since it must be present
        model = model_module.get_model(self.IMG_SIZE, self.NUM_CHANNELS_OUT,nrInputChannels)
        #Train model and get history of training performance
        model = self.train_model(model, train_gen, valid_gen, self.EPOCHS)     
        
        #Load model from directory
        #model = load_model("outDat/Unet_1_1.h5",custom_objects={"f1_m":utils.f1_m})
        #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[utils.f1_m])#Works with 2 classes as output from model.
        
        #Validate model and plot image, mask and prediction
        predictions, mean_dice_imagewise, total_dice_pixelwise = model_module.test_model(model, valid_gen)   # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        print('The imagewise dice similarity score is:', mean_dice_imagewise)
        print('The pixelwise dice similarity score is:', total_dice_pixelwise)

        #Plot predictions
        #plotting_module.plot_predictionsv2(predictions, validation_img_paths,validation_mask_paths,self.BATCH_SIZE)

    def train_model(self, model, train_generator, validation_generator, epochs):
        

        
        #Compile model dependant on output dimensions
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[utils.f1_m])#Works with 2 classes as output from model.
        model.summary()

        earlystopper = EarlyStopping(monitor='val_loss',patience=5, verbose=1,restore_best_weights=True)
        checkpointer = ModelCheckpoint('outDat/Unet_checkpoint_exp_17.h5', verbose=1, monitor='val_loss', save_best_only=True)
        callback = [earlystopper, checkpointer]

        # Train the model, doing validation at the end of each epoch.
        #Start timer
        t = time.time()
        history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callback)#Training with validation as generator or dataset
        #End timer
        elapsed = time.time()-t
        
        #Save model
        model.save('outDat/Unet_exp_17.h5')
                
        print("Runtime for training model, in hours: " , elapsed/(60*60))
        
        return model

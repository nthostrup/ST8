'''
Created on 31. mar. 2021

@author: Bruger
'''

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
        self.EPOCHS = 20
        self.IMG_SIZE = (512,512)
        self.NUM_CHANNELS_OUT = 1
    
    def run_main(self):
        #Load image and mask paths - training
        training_img_paths = utils.input_loader(self.TRAIN_INPUT_DIR)
        training_mask_paths = utils.mask_loader(self.TRAIN_MASK_DIR)
        
        #Load image and mask paths - validation
        validation_img_paths = utils.input_loader(self.VALID_INPUT_DIR)
        validation_img_paths = validation_img_paths[:140]#Shorten training time
        validation_mask_paths = utils.mask_loader(self.VALID_MASK_DIR)
        validation_mask_paths = validation_mask_paths[:140]#Shorten training time
        
        #Load image and mask paths - Test
        test_img_paths = utils.input_loader(self.TEST_INPUT_DIR)
        test_img_paths = test_img_paths[:140]#Shorten training time
        test_mask_paths = utils.mask_loader(self.TEST_MASK_DIR)
        test_mask_paths = test_mask_paths[:140]#Shorten training time
        
        #Print paths
        for input_path, target_path in zip(training_img_paths[:20], training_mask_paths[:20]):
            print(input_path, "|", target_path)
        
        train_gen = utils.make_generator(training_img_paths, training_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
        valid_gen = utils.make_generator(validation_img_paths, validation_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
        test_gen = utils.make_generator(test_img_paths, test_mask_paths, self.BATCH_SIZE, self.IMG_SIZE)#Randomizes lists inlist
        
        print("\n \n")
        for input_path, target_path in zip(training_img_paths[:5], training_mask_paths[:5]): #Check if lists randomized
            print(input_path, "|", target_path)
        
        #Plot of mask and image
        utils.plot_img_mask(validation_img_paths[2], validation_mask_paths[2])
        
        #Get model, input dimensions must match generator. Last dimension added since it must be present
        model = model_module.get_model(self.IMG_SIZE, self.NUM_CHANNELS_OUT)
        
        #Train model and get history of training performance
        history, model = model_module.train_model(model, valid_gen, test_gen, self.EPOCHS) #TODO: FEJLER I VALIDATION STEP
        
        #Load model from directory
        #model = load_model("Unet_MRI_1-2.h5")#Load
        
        #Validate model and plot image, mask and prediction
        predictions = model_module.test_model(model, test_gen, test_img_paths, test_mask_paths)
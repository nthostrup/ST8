'''
Created on 3. maj 2021

@author: Bruger
'''

from tensorflow.keras.models import load_model

#Import own modules
import utility_module as utils
import model_module
import math
import numpy as np
import time
import plotting_module
import MRI_generator_module
import random


def mainMethod(nrInputChannels, TEST_INPUT_DIR, TEST_MASK_DIR, TEST_SLICE_DIR):
        BATCH_SIZE = 30
        EPOCHS = 1
        IMG_SIZE = (512, 512)
        NUM_CHANNELS_OUT = 1
        SAMPLES_TO_RUN = 90
        

        #Load image and mask paths - Test
        test_img_paths = utils.input_loader(TEST_INPUT_DIR)
        test_img_paths = test_img_paths[:SAMPLES_TO_RUN]
        
        test_mask_paths = utils.mask_loader(TEST_MASK_DIR)
        test_mask_paths = test_mask_paths[:SAMPLES_TO_RUN]
        
        test_slice_paths = utils.input_loader(TEST_SLICE_DIR)
        test_slice_paths = test_slice_paths[:SAMPLES_TO_RUN]
        
        #To sort list of scans alphanumerically if one person is tested...
#         test_img_paths = sorted(test_img_paths, key=lambda item: int(item.partition('_')[2].partition('_')[2].partition('.')[0]))
#         test_mask_paths = sorted(test_mask_paths, key=lambda item: int(item.partition('_')[2].partition('_')[2].partition('.')[0]))
#         test_slice_paths = sorted(test_slice_paths, key=lambda item: int(item.partition('_')[2].partition('_')[2].partition('.')[0]))
        
               
        if(nrInputChannels == 1):
            test_gen = make_generator(test_img_paths, test_mask_paths, BATCH_SIZE, IMG_SIZE)#with 1 channel generator.Randomizes lists inlist
            random.Random(1337).shuffle(test_slice_paths)#Randomize slices also to match img and mask paths
        elif(nrInputChannels == 2):
            test_gen = make_generator_w_slicenumber(test_img_paths, test_mask_paths, BATCH_SIZE, IMG_SIZE,test_slice_paths)#with 2 channel generator.Randomizes lists inlist
        else:
            print("Input nr. of input channels!")
            return
        
        
        #Load model from directory
        model = load_model("C:/Users/Bruger/Desktop/Trained_Unets/Unet_prepr_3.h5",custom_objects={"f1_m":utils.f1_m})
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[utils.f1_m])#Works with 2 classes as output from model.
        
        #Validate model and plot image, mask and prediction
        predictions, mean_dice_imagewise, total_dice_pixelwise, f1_pr_image, volume_first_batch_pred, volume_first_batch_GT = test_model(model, test_gen)   # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        
        cleanedF1_pr_img = [x for x in f1_pr_image if str(x) != 'nan']
        
        
        
        
        print('The imagewise dice similarity score is: %1.6f' %mean_dice_imagewise)
        print('The pixelwise dice similarity score is: %1.6f' %total_dice_pixelwise)
        print('The volume of the first batch on prediction is: %1.3f L' %(volume_first_batch_pred))
        print('And volume of the first batch on ground truth is %1.3f L' %(volume_first_batch_GT))
        print('Which corresponds to: %3.1f pct. difference' %(100*(volume_first_batch_pred-volume_first_batch_GT)/volume_first_batch_GT))

        #Plot predictions
        #plotting_module.plot_predictionsv2(predictions, test_img_paths, test_mask_paths, BATCH_SIZE, test_gen)
        plotting_module.make_boxplot_plot(cleanedF1_pr_img)
        
        
def test_model(model,test_generator):
    batch_size = test_generator.batch_size
    all_samples = len(test_generator.mask_paths)
    num_batches = math.floor(all_samples/batch_size)  # floor forces it to round down
    
    print("Prediction in progres..")
    predictions = model.predict(test_generator)#, steps=n_batch) 
    print("Prediction finished!")
    
    #IMAGEWISE DICE
    f1_pr_batch = []
    f1_pr_image = []
    volume_batches = []
    volume_batches_GT = []
    for i in range(num_batches):
        _, mask = test_generator.__getitem__(i)
        mask = mask.astype('float32')  # typecast from uint8 to float32
        # faa kun 1 batch ind ad gangen
        prediction_batch = predictions[i*batch_size:(i+1)*batch_size]
        
        #Calculate imagewise dice
        f1_scores, f1_pr_image_batch = utils.dice_imagewise(y_true=mask, y_pred=prediction_batch)  # beregn f1 score for single prediction
        f1_pr_batch.append(f1_scores)       #Saves each dice for a batch
        f1_pr_image.append(f1_pr_image_batch) #saves each dice for each image
        
        #calculate volume
        tempVol_batch_pred, tempVol_batch_GT = calculate_volume_on_batch(prediction_batch,mask)
        volume_batches.append(tempVol_batch_pred)
        volume_batches_GT.append(tempVol_batch_GT)
        
    
    mean_dice_imagewise = np.mean(f1_pr_batch)
   
    # PIXELWISE DICE
    
    total_dice_pixelwise = utils.total_dice_pixelwise(predictions, num_batches, batch_size, test_generator) 

    return predictions, mean_dice_imagewise, total_dice_pixelwise, f1_pr_image[0], volume_batches[0], volume_batches_GT[0]

def calculate_volume_on_batch(prediction_batch, groundtruth_mask):
    pixelToArea = (0.8203*10**-3)*(0.8203*10**-3)#m^2 from dicom documentation and file
    depthOfPixel = 4*10**-3#4m
    
    
    volumePredicted = 0
    volumeGroundTruth = 0
    for i in range(len(prediction_batch)):
        pred = prediction_batch[i]
        pred = np.around(pred,0)
        
        mask = groundtruth_mask[i]
        
        uniquePred, countsPred = np.unique(pred, return_counts=True)
        uniqueMask, countsMask = np.unique(mask, return_counts=True)
        
        if(len(countsPred)==2):
            volumePredicted += (countsPred[1]*depthOfPixel*pixelToArea)*1000
        if(len(countsMask)==2):
            volumeGroundTruth += (countsMask[1]*depthOfPixel*pixelToArea)*1000
        
    
    
    return volumePredicted, volumeGroundTruth

def make_generator(input_img_paths, mask_paths, batch_size, img_size):
    
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(mask_paths)
           
    generator = MRI_generator_module.MRI_generator(batch_size, img_size, input_img_paths, mask_paths)
    
    
    return generator

#Makes a generator to feed network with tuple (input,masks) where input is TWO CHANNELS
def make_generator_w_slicenumber(input_img_paths, mask_paths, batch_size, img_size, input_img_slice_paths):
    
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(mask_paths)
    random.Random(1337).shuffle(input_img_slice_paths)
       
    generator = MRI_generator_module.MRI_generator_w_sliceNumber(batch_size, img_size, input_img_paths, mask_paths,input_img_slice_paths)
    
    
    return generator



#To use before final test
VALID_INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/Validation/Img"
VALID_MASK_DIR ="C:/Users/Bruger/Desktop/MRI/Validation/Mask"
VALID_SLICE_DIR ="C:/Users/Bruger/Desktop/MRI/Validation/SliceNr_validation"

VALID_INPUT_DIR_oneperson = "C:/Users/Bruger/Desktop/MRI/Validation/ImgOneperson"
VALID_MASK_DIR_oneperson ="C:/Users/Bruger/Desktop/MRI/Validation/MaskOneperson"
VALID_SLICE_DIR_oneperson ="C:/Users/Bruger/Desktop/MRI/Validation/SliceNr_validationOneperson"


TEST_INPUT_DIR = "C:/Users/Bruger/Desktop/MRI/Test/Img"
TEST_MASK_DIR ="C:/Users/Bruger/Desktop/MRI/Test/Mask"
TEST_SLICE_DIR ="C:/Users/Bruger/Desktop/MRI/Test/SliceNr_test"

nrInputChannels = 1 #Change dependant on the model to load whether it was trained with one or 2 input channels

t = time.time()
mainMethod(nrInputChannels, VALID_INPUT_DIR, VALID_MASK_DIR, VALID_SLICE_DIR)
#mainMethod(nrInputChannels, VALID_INPUT_DIR_oneperson, VALID_MASK_DIR_oneperson, VALID_SLICE_DIR_oneperson)
elapsed = time.time()-t
print("Runtime total, in hours: " , elapsed/(60*60))
print("Done")

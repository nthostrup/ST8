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
import matplotlib.pyplot as plt
import skimage.feature
import skimage.io
import os.path
from tensorflow.python.ops.gen_batch_ops import batch
#Global variables
SIZE_PR_MASK = []
IMG_SIZE = (512, 512)


def mainMethod(nrInputChannels, TEST_INPUT_DIR, TEST_MASK_DIR, TEST_SLICE_DIR):
        

        #Load image and mask paths - Test
        test_img_paths = utils.input_loader(TEST_INPUT_DIR)
        test_mask_paths = utils.mask_loader(TEST_MASK_DIR)
        test_slice_paths = utils.input_loader(TEST_SLICE_DIR)
        
        #Determines batch size and samples to run based on length of filelist
        if len(test_img_paths) < 50:
            BATCH_SIZE = len(test_img_paths)
            SAMPLES_TO_RUN = BATCH_SIZE
            batch_to_plot = 0
            onePersonRun = True
        else:
            BATCH_SIZE = 1
            SAMPLES_TO_RUN = 900
            batch_to_plot = 1
            onePersonRun = False
            
#         test_img_paths = test_img_paths[:SAMPLES_TO_RUN]
#         test_mask_paths = test_mask_paths[:SAMPLES_TO_RUN]
#         test_slice_paths = test_slice_paths[:SAMPLES_TO_RUN]
        
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
        model = load_model(r"C:\Users\Bruger\OneDrive\Dokumenter\Uni\Sundhedsteknologi\8. semester\Projekt\python\Trained_Unets/Unet_final_20210506-163849.h5",custom_objects={"f1_m":utils.f1_m})
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[utils.f1_m])#Works with 2 classes as output from model.
        
        #Validate model and plot image, mask and prediction
        predictions, mean_dice_imagewise, total_dice_pixelwise, f1_pr_batch_pr_image, volume_first_batch_pred, volume_first_batch_GT = test_model(model, test_gen)   # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        
        cleanedF1_pr_img_in_batch=[]
        for i in range(len(f1_pr_batch_pr_image)):
            cleanedF1_pr_img_in_batch.append([x for x in f1_pr_batch_pr_image[i] if math.isnan(x) == False])
            #Make boxplot for each batch
            #plotting_module.make_boxplot_plot(cleanedF1_pr_img_in_batch[i])
        
        #Write information to files:
        makeOutputfileImagewise([item for sublist in f1_pr_batch_pr_image for item in sublist],test_img_paths)
        #outputPredictions(predictions, test_gen, test_img_paths)
        
        #appendVolumeAndDiceToFile(volume_first_batch_GT, volume_first_batch_pred, mean_dice_imagewise, total_dice_pixelwise)
        
        #outputPredictionMasks(predictions,test_img_paths)
        
        print('The imagewise dice similarity score is: %1.6f' %mean_dice_imagewise)
        print('The pixelwise dice similarity score is: %1.6f' %total_dice_pixelwise)
        print('The volume of the first batch on prediction is: %1.3f L' %(volume_first_batch_pred))
        print('And volume of the first batch on ground truth is %1.3f L' %(volume_first_batch_GT))
        print('Which corresponds to: %3.1f pct. difference' %(100*(volume_first_batch_pred-volume_first_batch_GT)/volume_first_batch_GT))
        
        
        #Make scatter plot of mask proportion vs DICE
        #plotting_module.make_maskSize_vs_DICE_plot(f1_pr_batch_pr_image, SIZE_PR_MASK)
        
        
        #Make boxplot for all batches in one
        #plotting_module.make_boxplot_plot([item for sublist in cleanedF1_pr_img_in_batch for item in sublist])
        
        #Make boxplot for each batch
        #plotting_module.make_boxplot_plot(cleanedF1_pr_img_in_batch)
        #plt.show()

        #Plot predictions
        #plotting_module.plot_predictionsv2(predictions, test_gen, batch_to_plot)
        
        
        
def test_model(model,test_generator):
    batch_size = test_generator.batch_size
    all_samples = len(test_generator.mask_paths)
    num_batches = math.floor(all_samples/batch_size)  # floor forces it to round down
    
    print("Prediction in progres..")
    t = time.time()
    predictions = model.predict(test_generator)#, steps=n_batch) 
    
    elapsed = time.time()-t
    print("Prediction finished! Time: ",elapsed," sec")
    
    #IMAGEWISE DICE
    f1_pr_batch = []
    f1_pr_batch_pr_image = []
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
        f1_pr_batch_pr_image.append(f1_pr_image_batch) #saves each dice for each image
        
        #calculate volume
        tempVol_batch_pred, tempVol_batch_GT = calculate_volume_on_batch(prediction_batch,mask)
        volume_batches.append(tempVol_batch_pred)
        volume_batches_GT.append(tempVol_batch_GT)
        
    
    mean_dice_imagewise = np.mean(f1_pr_batch)
   
    # PIXELWISE DICE
    
    total_dice_pixelwise = utils.total_dice_pixelwise(predictions, num_batches, batch_size, test_generator) 

    return predictions, mean_dice_imagewise, total_dice_pixelwise, f1_pr_batch_pr_image, volume_batches[0], volume_batches_GT[0]

def calculate_volume_on_batch(prediction_batch, groundtruth_mask):
    pixSpacing = 0.78125
    pixelToArea = (pixSpacing*10**-3)*(pixSpacing*10**-3)#m^2 from dicom documentation and file
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
#             if(len(countsMask) == 1 & countsPred[1] <=1):
#                 SIZE_PR_MASK.append(0)#Placed here to count zero-GT with 
        if(len(countsMask)==2):
            volumeGroundTruth += (countsMask[1]*depthOfPixel*pixelToArea)*1000
            SIZE_PR_MASK.append(countsMask[1]/(512*512))
        elif(len(countsMask)==1):# & (len(countsPred)==1 | countsPred[1]<=1)):
            SIZE_PR_MASK.append(0)#Placed here to count zero-GT with 
    
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

def makeOutputfileImagewise(imgwise_dice,test_img_paths):
    directoryOut = "C:/Users/Bruger/OneDrive/Dokumenter/Uni/Sundhedsteknologi/8. semester/Projekt/ResultOutput/"
    filename = directoryOut +"NEW_ImagewiseDice_for_"+str(len(SIZE_PR_MASK))+"_images_GT0andPred0.csv"
    
    print("Saving ImgWise Dice to filename "+filename)
    
    np.savetxt(filename, 
           [(imgwise_dice), (SIZE_PR_MASK),(test_img_paths)],
           delimiter ="; ",
           newline='\n', 
           fmt ='%s')

#Only makes sense to use with a single person as input directory (otherwise volume is not correct)     
def appendVolumeAndDiceToFile(VolumeGT, VolumePredicted,ImgWiseDice,PixelWiseDice):
    directoryOut = "C:/Users/Bruger/OneDrive/Dokumenter/Uni/Sundhedsteknologi/8. semester/Projekt/ResultOutput/"
    filename = directoryOut +"ResultFile_pr_person_eksamen.csv"
    print("Saving person "+p+" to file..")
    f = open(filename,'a')
    
    np.savetxt(f, 
           [[str(VolumeGT).replace(".",","), str(VolumePredicted).replace(".",","), str(ImgWiseDice).replace(".",","), str(PixelWiseDice).replace(".",","), str(len(SIZE_PR_MASK))]],
           delimiter =";", 
           newline='\n',
           fmt ='%s')
    f.close()
    print("Save complete!")
    
#Method to plot mask on top of MR image
def outputPredictions(predictions, generator, test_img_paths):
    batch_size = generator.batch_size
    
    print("Creating masks and saving images")
    for batch_to_show in range(len(predictions)//batch_size):
        mri_img, mask = generator.__getitem__(batch_to_show) #getitem input is batchindex.
        
        for i in range(batch_size):
            # load predicted mask
            predicted_mask = predictions[i+batch_size*batch_to_show]
            imgName = os.path.basename(test_img_paths[i+batch_size*batch_to_show])
            rounded = np.round(predicted_mask,0) #rounds the array to integers.
            
            # load MRI        
            mr_img_arr = mri_img[i] #Added to plot preprocessed MRI
            
            # load ground truth mask
            mask_arr = mask[i]#ADDED to plot preprocessed MRI
            
            # beregner edges for ground thruth masken
            edges_gt = skimage.feature.canny(
            image=np.squeeze(mask_arr),
            sigma=1,
            low_threshold=0.1,
            high_threshold=0.9,
            ) #shape = 512,512
            # beregner edges for den predicterede maske
            edges_pred = skimage.feature.canny(
            image=np.squeeze(rounded),
            sigma=1,
            low_threshold=0.1,
            high_threshold=0.9,
            ) #shape = 512,512
            # rgb for roed er (255,0,0)
            #laver ground truth maske i roede farver.
            Mask_gt_pred = np.zeros((512,512,3))
            for l in range(0,512):
                for w in range(0,512):
                        if edges_gt[l,w]==True:
                            Mask_gt_pred[l,w,:] = [1,0,0] #red
            
            
            for l in range(0,512):
                for w in range(0,512):
                    if edges_pred[l,w]==True:
                        Mask_gt_pred[l,w,:] = [0,1,0] #green
            # Output img to file
            pic = np.maximum(mr_img_arr,Mask_gt_pred)  
            
            directoryOut = "C:/Users/Bruger/OneDrive/Dokumenter/Uni/Sundhedsteknologi/8. semester/Projekt/ResultOutput/predictedMasksImg_eksamen/IBS/"
            filename = directoryOut + imgName
            
            skimage.io.imsave(filename, pic)
            
    print("All images saved")
    
def outputPredictionMasks(predictions, test_img_paths):
        for i in range(len(predictions)):        
            predicted_mask = predictions[i]
            rounded = np.round(predicted_mask,0) #rounds the array to integers.    
            
            imgName = os.path.basename(test_img_paths[i])

            directoryOut = "C:/Users/Bruger/OneDrive/Dokumenter/Uni/Sundhedsteknologi/8. semester/Projekt/ResultOutput/predictedMasksMask_eksamen/IBS/"+p+"/"
            filename = directoryOut + imgName
            
            skimage.io.imsave(filename, rounded)

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
for i in range(1,2):
    SIZE_PR_MASK = []
    p = "Person"+str(i)
    TEST_INPUT_DIR_oneperson = "C:/Users/Bruger/Desktop/MRI/Eksamen_MRI_billeder/img_IBS/ImgPersons/"+p
    TEST_MASK_DIR_oneperson ="C:/Users/Bruger/Desktop/MRI/Eksamen_MRI_billeder/img_IBS/MaskPersons/"+p
    TEST_SLICE_DIR_oneperson ="C:/Users/Bruger/Desktop/MRI/Test/SliceNr_testPersons/"+p
    
    
    nrInputChannels = 1 #Change dependant on the model to load whether it was trained with one or 2 input channels
    
    t = time.time()
    mainMethod(nrInputChannels, TEST_INPUT_DIR, TEST_MASK_DIR, TEST_SLICE_DIR)#For imagewise DSC (entire folder)
    #mainMethod(nrInputChannels, TEST_INPUT_DIR_oneperson, TEST_MASK_DIR_oneperson, TEST_SLICE_DIR_oneperson) #For one person at a time
    
    #mainMethod(nrInputChannels, VALID_INPUT_DIR, VALID_MASK_DIR, VALID_SLICE_DIR)
    #mainMethod(nrInputChannels, VALID_INPUT_DIR_oneperson, VALID_MASK_DIR_oneperson, VALID_SLICE_DIR_oneperson)
    elapsed = time.time()-t
    print("Runtime total, in hours: " , elapsed/(60*60), "for person ", p)
    plt.show()
print("Done")

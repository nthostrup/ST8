'''
Created on 21. apr. 2021

@author: Bruger
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import skimage.feature
from itertools import count
from Tools.scripts.objgraph import flat

#Helper to plot image and mask pair
def plot_img_mask(img_path,mask_path):
    
    img = load_img(img_path)
    img_arr = img_to_array(img)
    img_arr /= 255.
    
    plt.figure()
    plt.imshow(img_arr,cmap='gray')
    
       
    #maskImg=load_img(mask_path, color_mode="grayscale")
    #maskImg=np.expand_dims(maskImg, 2)
    
    maskImg=plt.imread(mask_path)
    plt.figure()
    plt.imshow(maskImg,cmap='gray')
    
    plt.show()

def plot_training_history(history):
    
    #Plot performance from training
    #TODO: Consider plotting dice measure
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    #plt.plot(epochs, acc, 'bo', label='Training acc')
    #plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.legend()
    
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show(block=False)

#Method to plot mask on top of MR image
def plot_predictionsv2(predictions, generator,batch_to_show):
    
    mri_img, mask = generator.__getitem__(batch_to_show) #getitem input is batchindex.
    batch_size = generator.batch_size
    for i in range(batch_size):
        # load predicted mask
        predicted_mask = predictions[i+batch_size*batch_to_show]
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
                        Mask_gt_pred[l,w,:] = [255,0,0] #red
        
        
        for l in range(0,512):
            for w in range(0,512):
                if edges_pred[l,w]==True:
                    Mask_gt_pred[l,w,:] = [0,255,0] #green
        # plot mr billede
        plt.figure()
        pic = mr_img_arr+Mask_gt_pred
        plt.imshow(pic)
        plt.title('MRI, ground thruth is red and prediction is green')

    plt.show()    
def plot_predictions(predictions, input_img_paths, mask_paths):
    #Selected slice to compare
    i = 5;
    #Plots 
    #plt.subplot(1,2,1)
    predicted_mask = predictions[i]
    rounded = np.round(predicted_mask,0)
    plt.figure()
    plt.imshow(rounded,cmap='gray')
    
    #plt.subplot(1,2,2)
    #Print paths to ensure that they match
    print("input path: " + input_img_paths[i])
    print("mask path: " + mask_paths[i])
    
    #Plot MRI and mask via utility module
    plot_img_mask(input_img_paths[i], mask_paths[i])
    
    #Plot mask alone
    #maskImg=plt.imread(mask_paths[i])
    #plt.imshow(maskImg,cmap='gray')
    #plt.show()
    
def make_boxplot_plot(dice):
    plt.figure()
    plt.boxplot(np.asarray(dice))
    plt.title('Boxplot of DICE pr. image. Total images: %d'%(len(dice)))
    plt.xlabel("Batch nr.")
    plt.ylabel("Dice score")
    plt.ylim(0,1)
    #plt.show()

def make_maskSize_vs_DICE_plot(dice,maskSize):
    flattenDice = [item for sublist in dice for item in sublist]
    
    plt.figure()
    plt.scatter(maskSize,flattenDice)
    plt.title('Plot GT mask size vs DICE pr. image. Total batches: %d . Total images: %d'%(len(dice),len(flattenDice)))
    plt.xlabel("Mask size, in proportion of total img")
    plt.ylabel("Dice score")
    
    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(maskSize,flattenDice, 1)
    
    #add linear regression line to scatterplot 
    plt.plot(maskSize, m*np.array(maskSize)+b, color="red")
    textStr = "Regression: y=%2.3f x + %1.3f"%(m,b)
    plt.text(0.03, 0, textStr)
    
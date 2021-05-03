'''
Created on 25. mar. 2021

@author: Bruger
'''
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#Import own modules
import preprocessing_module as pp_module

#Class inherited from generic Keras class, which is a generator
class MRI_generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, mask_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.mask_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.mask_paths[i : i + self.batch_size]
        
        #TODO: Fix image size in numpy array and datatype
        #TODO: Should variable img be used directly? In other words shohuld labels be vectors and samples be images?
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32") #+(3,) since img_size is the same for both input and label
        for j, path in enumerate(batch_input_img_paths):
            #print(path)
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img_arr = img_to_array(img)
            img_arr /= 255.
            
            ## ::::PREPROCESSING of MRI input::::
            img_arr = pp_module.CLAHE(img_arr)
            
            
            x[j] = img_arr
            
        
        
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8") #+(1,) since img_size is the same for both input and label       
        for j, path in enumerate(batch_target_img_paths):
            #print(path)
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                        
            img_arr = img_to_array(img)
            img_arr /= 255.
            img_arr = np.around(img_arr)# "Binarize masks to be either 0 or 1
            
            
            ##::::::PREPROCESSING OF MASK:::::
            img_arr = pp_module.openClose(img_arr)
            
            y[j] = img_arr
            
            # Ground truth labels are 0, 1. Divide by 255 to get this range
            
            
        return x, y
    
    #Class inherited from generic Keras class, which is a generator
class MRI_generator_w_sliceNumber(MRI_generator):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, mask_paths, input_img_slice_paths):
        super().__init__(batch_size, img_size, input_img_paths, mask_paths)
        self.input_img_slice_paths = input_img_slice_paths

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_input_img_slice_paths = self.input_img_slice_paths[i : i + self.batch_size]
        batch_target_img_paths = self.mask_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="float32") #+(2,) since we use 2 channels
        for j, (path, slice_path) in enumerate(zip(batch_input_img_paths,batch_input_img_slice_paths)):
            #print(path)
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img_arr = img_to_array(img)
            img_arr /= 255.
            
            ## ::::PREPROCESSING of MRI input::::
            img_arr = pp_module.CLAHE(img_arr)
            
            
            slice_img = load_img(slice_path, target_size=self.img_size, color_mode="grayscale")
            slice_arr = img_to_array(slice_img)
            slice_arr /= 255.
            
            
            
            x[j] = np.concatenate((img_arr,slice_arr),2)
            
        
                
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8") #+(1,) since img_size is the same for both input and label       
        for j, path in enumerate(batch_target_img_paths):
            #print(path)
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                        
            img_arr = img_to_array(img)
            img_arr /= 255.
            img_arr = np.around(img_arr)# "Binarize masks to be either 0 or 1
            
            img_arr = pp_module.openClose(img_arr)
            
            y[j] = img_arr
            
            # Ground truth labels are 0, 1. Divide by 255 to get this range
            
            
        return x, y
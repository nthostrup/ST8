'''
Created on 25. mar. 2021

@author: Bruger
'''
import numpy as np


#IMport own modules
import main_module
import utility_module as utils

#Global variables/hyperparameters
data_path = '/data/'
TRAIN_INPUT_DIR = data_path + "Training/Img"
TRAIN_MASK_DIR = data_path + "Training/Mask"
VALID_INPUT_DIR = data_path + "Validation/Img"
VALID_MASK_DIR = data_path + "Validation/Mask"
TEST_INPUT_DIR = data_path + "Test/Img"
TEST_MASK_DIR = data_path + "Test/Mask"

mainObject = main_module.main_class(TRAIN_INPUT_DIR,TRAIN_MASK_DIR,VALID_INPUT_DIR, VALID_MASK_DIR, TEST_INPUT_DIR, TEST_MASK_DIR)

mainObject.run_main()


print("Done")

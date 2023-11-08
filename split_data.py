import time
import os
import shutil 
import random
import glob

import datacleaning.tkinterSetup as tks
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

#select images and split
def selectImages_toSplit():
    count = 0
    run = True
    while(run):
        images = []
        print("select the images to split into directories")
        images = tks.select_files()
        print(f'images selected {images}')
        if len(images) == 0:
            print("NO IMAGES SELECTED")
            return
        while(run):
            splitString = input("what percent split (test/train) ENTER digit or decimal")
            if(splitString.isnumeric() == False):
                print("enter a number")
                continue
            split = int(splitString)
            if(split < 0):
                print("enter positive number")
                continue
            if(split<1 and split > 0):
                split = split * 100
            run = False
    trainSplit = 100 - split
    start_time = time.time()
    trainCount = 0
    for sel_image in images:
        print(f"{len(images)} selected images")
        if os.path.isfile(sel_image):
            path = sel_image
            if trainCount <= trainSplit:
                shutil.move(path, '/home/autodrivex/imageProcessing/resNet/train')
                trainCount += trainCount
                continue
            shutil.move(path, '/home/autodrivex/imageProcessing/resNet/train')
        print(f"{count} images processed")
        print("--- %s seconds ---" % (time.time() - start_time))
        runin = input("augment another batch? (y or n)")
        if runin != 'y':
            run = False
if __name__ == '__main__':
    selectImages_toSplit()
# #randomly select images from directory
# def randomlySelectImages_toSplit():
#     os.chdir('colors')
#     #make directories for train, test, validation data sets

#     #move subsets of images into newly created directories
#     os.chdir('train/green')
#     for i in random.sample(glob.glob('*.jpg'), 150):
#         shutil.move(i, '../../test/green') 
#     os.chdir('../../train/green')     
#     for i in random.sample(glob.glob('*.jpg'), 75):
#         shutil.move(i, '../../valid/green')
#     os.chdir('../../train/yellow')
#     for i in random.sample(glob.glob('*.jpg'), 150):
#         shutil.move(i, '../../test/yellow')    
#     os.chdir('../../train/yellow')    
#     for i in random.sample(glob.glob('*.jpg'), 75):
#         shutil.move(i, '../../valid/yellow')
#     os.chdir('../../train/red')   
#     for i in random.sample(glob.glob('*.jpg'), 150):
#         shutil.move(i, '../../test/red')  
#     os.chdir('../../train/red')    
#     for i in random.sample(glob.glob('*.jpg'), 75):
#         shutil.move(i, '../../valid/red')

'''
file showing how to augment an image multiple ways to create a new set of images
allows you to make get more data quickly
good solution to overfitting
'''
from email import message
from fileinput import filename
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


#%matplotlib inline

#plot image function 
#https://www.tensorflow.org/tutorials/images/classification#visualize_training_images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#augment random image
def augmentRadomBatch():
    #Specify how many photos to select and augment
    numString = input("enter number of photos to augment: ")
    numToAugment = int(numString)
    colorString = input("enter color light to add (red, green, yellow):")
    colorToAugment = colorString
    for x in range(numToAugment):
        #creates images with the specified parameters
        #https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.1, zoom_range=0.05, 
            channel_shift_range=2, horizontal_flip=True)

        #filename of randomly chosen dog image 
        chosen_image = random.choice(os.listdir(f'data/train/{colorToAugment}/'))
        #create path to image chosen above
        image_path = f'data/train/{colorToAugment}/' + chosen_image
        #the image itself from path as np array
        image = np.expand_dims(plt.imread(image_path),0)
        #show the selected image
        plt.imshow(image[0])
        #generate batch of augmented images
        #aug_iter = gen.flow(image)
            #save the augmented data
        aug_iter = gen.flow(image, save_to_dir=f'data/train/{colorToAugment}/', save_prefix='aug-data-', save_format='jpeg')
        #get 10 samples of augmented image
        aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    #show the augmented images
    plotImages(aug_images)

def augmentSpecificImages():
    run = True
    while(run):
        colorString = input("enter color light to add (red, green, yellow): ")
        colorToAugment = colorString
        chosen_image = input("enter name of photo to augment: ")
        image_path = f'data/train/{colorToAugment}/' + chosen_image
        if os.path.isfile(image_path):
            gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.1, zoom_range=0.05, 
                    channel_shift_range=2, horizontal_flip=True)
            image = np.expand_dims(plt.imread(image_path),0)
            plt.imshow(image[0])   
            aug_iter = gen.flow(image, save_to_dir=f'data/train/{colorToAugment}/', save_prefix='aug-data-', save_format='jpeg')
            aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
            plotImages(aug_images)
            runin = input("augment another batch? (y or n)")
            if runin != 'y':
                run = False
        else:
            print("image not found try again (ctrl+c to quit)")

#select images to augment from file explorer 
#this is better than the other functions 

import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

def select_filesOF():
    '''
    there are a lot of things to do with tkinter
    source:https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/
    '''
    pass
   

def select_files():
    filetypes = (
        ('jpg files', '*.jp*'),
        ('png files', '*.png'),
        ('All files', '*.*')
    )

    filenames = fd.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filenames
    )
    return filenames

    
def augmentImageOF():
    run = True
    while(run):
        images = []
        print("select images to augment")
        images = select_files()
        print(f'images selected {images}')
        print('choose folder to save augment images')
        folder_path = fd.askdirectory()
        for sel_image in images:
            if os.path.isfile(sel_image):
                gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.1, zoom_range=0.05, 
                        channel_shift_range=2, horizontal_flip=True)
                image = np.expand_dims(plt.imread(sel_image),0)
                plt.imshow(image[0])   
                aug_iter = gen.flow(image, save_to_dir=folder_path, save_prefix='aug-data-', save_format='jpg')
                aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(50)]
                plotImages(aug_images)
        runin = input("augment another batch? (y or n)")
        if runin != 'y':
            run = False

if __name__ == '__main__':
    augmentImageOF()
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import adam_v2
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
print(tf.config.list_physical_devices('GPU'))
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

# Organize data into train, valid, test dirs
os.chdir('data')
#make directories for train, test, validation data sets
if os.path.isdir('test/green') is False:
    os.makedirs('test/green')
    os.makedirs('test/red')
    os.makedirs('test/yellow')
    os.makedirs('valid/green')
    os.makedirs('valid/red')
    os.makedirs('valid/yellow')
    #move subsets of images into newly created directories
    os.chdir('train/green')
    for i in random.sample(glob.glob('*.jpg'), 150):
        shutil.move(i, '../../test/green') 
    os.chdir('../../train/green')     
    for i in random.sample(glob.glob('*.jpg'), 75):
        shutil.move(i, '../../valid/green')
    os.chdir('../../train/yellow')
    for i in random.sample(glob.glob('*.jpg'), 150):
        shutil.move(i, '../../test/yellow')    
    os.chdir('../../train/yellow')    
    for i in random.sample(glob.glob('*.jpg'), 75):
        shutil.move(i, '../../valid/yellow')
    os.chdir('../../train/red')   
    for i in random.sample(glob.glob('*.jpg'), 150):
        shutil.move(i, '../../test/red')  
    os.chdir('../../train/red')    
    for i in random.sample(glob.glob('*.jpg'), 75):
        shutil.move(i, '../../valid/red')

os.chdir('../')
#set path variables to appropiate locations
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

#Create DirectoryIterator objects to store data
#create batches from the train, valid, and test directories 
#ImageDataGenerator(preprocess function).flow_from_directory(path, resize size, image classes, batchsize)
#NOTE: preprocessing function skews color data
'''
VGG16 whitepaper: https://arxiv.org/pdf/1409.1556.pdf
"The only preprocessing we do is subtracting the mean RGB value, 
computed on the training set, from each pixel."
'''
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['yellow', 'red', 'green'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['yellow', 'red', 'green'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['yellow', 'red', 'green'], batch_size=10, shuffle=False)

#generate a batch of images and labels from training set
imgs, labels = next(train_batches)
#plot function (https://www.tensorflow.org/tutorials/images/classification#visualize_training_images)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    
    plotImages(imgs)
    print(labels)


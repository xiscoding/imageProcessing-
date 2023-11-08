'''
organize photos into red green and yellow folders
RESNET OUT TENSOR FORMAT # [green, red, yellow]
'''

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import cv2
import time
import os
import shutil 

import tkinterSetup as tks
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import datetime


print("cuda?: ", torch.cuda.is_available())

#cv2.imshow('trafficLight', img_msg)

model = models.resnet18(pretrained=True)

class TrafficLight: 
    def __init__(self):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
        #load network
        model.load_state_dict(torch.load("/home/autodrivex/imageProcessing/resNet/resnet8_28_22.pth"))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
    
        
    def image_loader(loader, image_name):
        image = PIL.Image.open(image_name)
        image = loader(image).float()
        image = torch.tensor(image, requires_grad=True)
        image = image.unsqueeze(0)
        #if the model is trained on gpu
        #if the model is not trained on gpu remove line below
        #https://stackoverflow.com/questions/62302878/input-type-torch-floattensor-and-weight-type-torch-cuda-floattensor-should-b
        #https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
        image = image.cuda()
        return image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#mean and std from ImageNet

    ])
    def make_prediction(self, path):
        model.eval()
        rawResults = model(TrafficLight.image_loader(TrafficLight.transform, path))
        prediction = torch.maximum(rawResults, torch.tensor((1,1,1)).cuda())
        print(f'raw result: {rawResults}')
        print(prediction.shape)
        print(torch.argmax(prediction).item())
        return torch.argmax(prediction).item()

def make_decision(result_index, path):
    greenfolder = '/home/autodrivex/imageProcessing/colors/1015_vid2/green'
    redfolder = '/home/autodrivex/imageProcessing/colors/1015_vid2/red'
    yellowfolder = '/home/autodrivex/imageProcessing/colors/1015_vid2/yellow'
    if result_index == 0:
        print('Green')
        print(f"image path: {path}")
        
        x = datetime.datetime.now()
        split = path.split('.')
        path2 = split[0] + x.strftime("%f") + f".{split[1]}"
        print(path2)
        os.rename(path, path2)
        shutil.move(path2, greenfolder)
    if result_index == 1:
        print('Red')
        print(f"image path: {path}")
        x = datetime.datetime.now()
        split = path.split('.')
        path2 = split[0] + x.strftime("%f") + f".{split[1]}"
        os.rename(path, path2)
        shutil.move(path2, redfolder)
    if result_index == 2:
        print('Yellow')
        print(f"image path: {path}")
        x = datetime.datetime.now()
        split = path.split('.')
        path2 = split[0] + x.strftime("%f") + f".{split[1]}"
        os.rename(path, path2)
        shutil.move(path2, yellowfolder)
    else:
        print(result_index)
        print(f"image path: {path}")
def selectImagesTemplate():
        count = 0
        run = True
        while(run):
            images = []
            print("select images to augment")
            images = tks.select_files()
            print(f'images selected {images}')
            
            start_time = time.time()
            for sel_image in images:
                if os.path.isfile(sel_image):
                    path = sel_image
                    
                    print("Image Received at {}".format(path))
                    result_index = TrafficLight().make_prediction(path)
                    make_decision(result_index, path)
                    count = count + 1
            print(f"{count} images processed")
            print("--- %s seconds ---" % (time.time() - start_time))
            runin = input("augment another batch? (y or n)")
            if runin != 'y':
                run = False

if __name__ == '__main__':
    import time
    
    selectImagesTemplate()
    # [green, red, yellow]

    # thinks close ups of tl is yellow
    # far away with full road is red
    # green is good
    
   
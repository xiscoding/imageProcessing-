from statistics import mean
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

#to ensure that only one OpenMP is running
#source: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#visualize data

# root = tk.Tk()
# root.title('Tkinter File Dialog')
# root.resizable(False, False)
# root.geometry('300x150')


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
        title='Selected Files',
        message=filenames
    )
    return filenames

# open_button = ttk.Button(
#     root,
#     text='Open Files',
#     command=select_files
# )
img_list = []
mean_list = [] 
std_list = [] 

#use tkinter to open file explore and select images
    #convert each image into a np array
    #append image to img_list
#input: none
#output: none img_list is filled with np array images
def collect_data():
    print('ues gui to select images to normalize: ')
    # open_button.pack(expand=True)
    # root.mainloop()
    selected_images = select_files()
    for sel_image in selected_images:
        if os.path.isfile(sel_image):
           img_path = sel_image
           img = Image.open(img_path)
        #    img_np = transform_toTensor(img)
           img_np = np.array(img)
           img_list.append(img_np)
        else:
            print("image path invalid")

#tensor-fy data
#input: single image
#output: single image in tensor form
def transform_toTensor(img):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tr = transform(img)
    img_np = np.array(img_tr)
    return img_np

def transform_list_toTensor(list):
    tensor_list = [] 
    for im in list:
        im_tensor = transform_toTensor(im)
        tensor_list.append(im_tensor)
    return tensor_list


#get distribution of tensor-fyd pixels
#input: list of images 
#output: histogram 
def get_pixelDistribution(img_list):
    # plt.hist(img_np.ravel(), bins=50, density=True)
    img_list_squeeze = np.squeeze(img_list)
    bin_count = int(len(img_list)/5)
    plt.hist(img_list_squeeze, bins=bin_count, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    # mean, std = img_list_squeeze.mean(), img_list_squeeze.std()
    plt.show()

def get_meanDistribution(mean_list):
    # plt.hist(img_np.ravel(), bins=50, density=True)
    bin_count = int(len(mean_list))
    plt.hist(mean_list, bins=bin_count, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    # mean, std = img_list_squeeze.mean(), img_list_squeeze.std()
    plt.show()


#determine mean and std of tensor-fyd data
    #iterate through img_list 
        #caluate mean of each image
        #return list of means
def get_means(img_list):
    for im in img_list:
        im = transform_toTensor(im)
        im_mean = np.mean(im)
        mean_list.append(im_mean)
    print(mean_list)


#main function
if __name__ == '__main__':
    collect_data()
    get_means(img_list)
    get_meanDistribution(mean_list)
    # print(img_list)
    # get_pixelDistribution(img_list)
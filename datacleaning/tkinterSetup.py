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

# def selectImagesTemplate():
#     run = True
#     while(run):
#         images = []
#         print("select images to augment")
#         images = select_files()
#         print(f'images selected {images}')
#         print('choose folder to save augment images')
#         folder_path = fd.askdirectory()

            
#         runin = input("augment another batch? (y or n)")
#         if runin != 'y':
#             run = False

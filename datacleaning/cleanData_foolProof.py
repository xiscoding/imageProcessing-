'''
pick a directory and clean the data
source for tkinter code: https://stackoverflow.com/questions/66663179/how-to-use-windows-file-explorer-to-select-and-return-a-directory-using-python
'''
import cleanData as cld
import tkinter
from tkinter import filedialog



def cleanData():
    print("pick the folder you want to clean")
    tkinter.Tk().withdraw()
    folder_path = filedialog.askdirectory()
    print(folder_path)
    print("count the number of files in selected folder(press c)")
    print("delete all balck images (press b)")
    print("delete specific image (press s)")
    print("delete all images in a directory (press A)")

    choice = input("make your choice then press enter: ")

    if choice == 'c':
        cld.countFiles(folder_path)
    elif choice == 'b':
        cld.deleteblackImages(folder_path)
    elif choice == 's':
        cld.deleteSingleImage(folder_path)
    elif choice == 'A':
        cld.deleteAllindir(folder_path)

if __name__ == '__main__':
    run = True
    while(run):
        cleanData()
        if input('to quit enter(q): ') == 'q':
            run = False
            print("quitting now")
            break
        print("get ready to select another folder to clean")
from email.mime import image
import os
import cv2
import shutil

def countFiles(dir_path):
    count = 0 
    for root, dirs, files in os.walk(dir_path):
        for filename in files: 
            print(os.path.join(root, filename))
            count += 1
    print(count)
#delete all black images
def deleteblackImages(dir_path):
    black_count = 0
    #loop through folder
    #if image is black move to new subfolder 
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            gray_version = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(gray_version) == 0:
                print("Image is all black")
                shutil.move(image_path, 'allBlackImgs')
                black_count += 1
            else:
                print("Image is fine")
    print(f"{black_count} images moved to folder allblackImgs")
#delete specific images
def deleteSingleImage(dir_path):
    colorString = input("enter color light to remove (red, green, yellow): ")
    colorToAugment = colorString
    chosen_image = input("enter name of photo to remove: ")
    image_path = f'data/train/{colorToAugment}/' + chosen_image
    if os.path.isfile(image_path):
        os.remove(image_path)
        print(f"{image_path} has been deleted")
    else:
        print(f"{image_path} does not exist")
#delete subset of images
def deleteAllindir(dir_path):
    if input('this is going to move ALL FILES into trash... you sure?(y)') == 'y':
        file_count = 0
        #delete every file in specified directory 
        for root, dirs, files in os.walk(dir_path):
            for filename in files: 
                image_path = os.path.join(root, filename)
                os.remove(image_path)
                print(f"{image_path} has been deleted")
                file_count += 1
        print(f'{file_count} files have been deleted')
    else:
        print('ok no deletions will be made goodbye')

def deleteDuplicates():
    pass

def rotateImages(dir_path):
    # dir_path = input('input directory with images to flip')
    file_count = 0
    #delete every file in specified directory 
    for root, dirs, files in os.walk(dir_path):
        for filename in files: 
            image_path = os.path.join(root, filename)
            src = cv2.imread(image_path)
            rotated = cv2.rotate(src, cv2.ROTATE_180)
            cv2.imshow("Rotated by 90 Degrees", rotated)
            cv2.waitKey(1)
            print(f"{image_path} has been rotated")
            cv2.imwrite(image_path, rotated)
            file_count += 1
    print(f'{file_count} files have been rotated')


if __name__ == '__main__':
    if input('clean data (y)') == 'y':
        dir_to_clean = input('enter folder to clean (test, train, valid): ')
        color_to_clean = input('enter color to clean (red, green, yellow): ')
        dir_path = f'data/{dir_to_clean}/{color_to_clean}'
    else:
        dir_path = input('enter directory path to clean ')
    #iterate over files in the directory
    
    
    print("delete all balck images (press B)")
    print("delete specific image (press S)")
    print("delete all images in a directory (press A)")
    print("rotate images (press R)")
    choice = input("make your choice then press enter")

    if choice == 'B':
        deleteblackImages()
    if choice == 'S':
        deleteSingleImage()
    if choice == 'A':
        deleteAllindir()
    if choice == 'R':
        rotateImages(dir_path)
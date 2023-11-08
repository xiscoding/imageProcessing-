import cv2
import numpy
vidcap = cv2.VideoCapture('/home/autodrivex/imageProcessing/videos/Photos-1015/VID_20221015_122737733.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/vids_1015/vid3/frame%d_vid3.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
print(f"{count} frames caputured")
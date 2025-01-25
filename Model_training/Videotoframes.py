import os
import random

import numpy
import cv2
from glob import glob

'''
vidcap = cv2.VideoCapture("VID_20221016_134858.mp4")
success,image = vidcap.read()
count = 876
while success and count < 878:
  cv2.imwrite("Nakretka%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
'''


# Losowa numeracja zdjec
import os
folder = 'Widelec/'
l = os.listdir(folder)
import random
numbers = []
for i in range(0,len(l)):
    numbers.append(i)
random.shuffle(numbers)

for i in range(0,len(l)):
    print(i)
    os.rename(folder+l[i], folder+'Widelec-'+str(numbers[i])+'.jpg')


# Generator obrazow wejsciowych
from PIL import Image
import os
l = os.listdir("train_clearbckgrd")

foreground = Image.open("Puszka853.png")
foreground = foreground.resize((450, 450))


last_wallpaper = 3000
print(l[0])
col = 8
row = 5

for j in range(0, row):
    print("j",j)
    for i in range(0+last_wallpaper,col+last_wallpaper):
        # print(j*col+i)
        background = Image.open("train_clearbckgrd/"+l[j*col+i])
        background = background.resize((900, 900))
        temp_background = background.copy()
        temp_background.paste(foreground.rotate(i*72), (300, 300), foreground.rotate(i*72))
        #temp_background.show()
        temp_background.save("Puszka"+str(j*col+i)+".jpg")
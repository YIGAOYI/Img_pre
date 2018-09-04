import numpy as np
#import Image
from PIL import Image
import random
import cv2
import os

# raw_mg to 256 * 256  randomly
def box(): 

# initialize
    height = 540  # 540
    width  = 960  # 960

    crop_h = 256
    crop_w = 256

# random range
    cols = height - crop_h + 1 # 258
    rows = width - crop_w + 1 # 705

    h = random.randint(0,cols)
    w = random.randint(0,rows)

    y = h+crop_h
    x = w+crop_w

    box = (w, h, x, y)
#   image = image.crop(box)
    return box


file_dir = "/home/gy/fat_s/test/"
classes = {"master"}

box_m = box()

for index,name in enumerate(classes):
    class_path = file_dir+name+"/"
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    #for image_name in os.listdir(class_path):
    path_list=os.listdir(class_path)
    path_list.sort() #对读取的路径进行排序
    for image_name in path_list: 
        img_path = class_path + image_name #读取每一个图片路径
        print(img_path)
        count = 0
        image = cv2.imread(img_path)
        if count >= 1 :
            box_m = box()
            print(box_m)
            count = 0
        # Omit seg image image_name by its name
        if 'json' in image_name:
            continue
 
        if image_name.find('depth') >= 0:
            image =  image[box_m[1]:box_m[3], box_m[0]:box_m[2]]
            #image = image.crop(box)

        elif 'jpg' in image_name:
            image =  image[box_m[1]:box_m[3], box_m[0]:box_m[2]]
            #image = image.crop(box)
            
        else:
            image =  image[box_m[1]:box_m[3], box_m[0]:box_m[2]]
            count =+ 1
            print (count)
        cv2.imwrite('/home/gy/fat_s/after/'+image_name,image)
       


#img1 = crop_ran(image)
#img1.show()
#img1.save("lena2.jpg")


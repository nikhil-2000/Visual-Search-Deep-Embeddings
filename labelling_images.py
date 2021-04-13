"""
Plan:
Read in first image of each folder with folder name

Then for each image, output the image, user provides short input for label
Store this label in dict where file name maps to label
"""
import glob
import os
import subprocess

from tqdm import tqdm
from PIL import Image
import cv2

labels = ['Trouser',
          'Dress',
          'Coat',
          'Bag',
          'Sandal',
          'Trainer',
          'Smart shoes',
          'Hat',
          'Wrist wear',
          'Accessory',
          'Glasses',
          'Sun glasses',
          'Ear ring',
          'Wallet/Purse',
          'Briefs',
          'Long sleeved top',
          'Ladies pants',
          'Shorts',
          'Short sleeved top',
          'Necklace',
          'Belt',
          'Skirt']
images_path = "../uob_image_set"


def full_path(dir_name):
    image_folder = os.path.join(images_path, dir_name)

    img_name = os.listdir(image_folder)[0]
    return os.path.join(image_folder, img_name)

def output_labels():
    for i, lbl in enumerate(labels):
        print(i,".",lbl)

def get_images():
    folders = os.listdir(images_path)[:20]
    print("Loading Images...")
    paths = [full_path(folder) for folder in tqdm(folders)]

    print("LOADED")
    return paths, folders


def label_image(image_path, name, wait_time = 0):
    #If wait time is 0, then you have to click a key to close the window

    img = cv2.imread(image_path)
    img = cv2.resize(img, ( 500, 667))
    cv2.imshow(name, img)
    cv2.waitKey(wait_time)
    cv2.destroyWindow(name)
    label = input("Write number to represent label: ")
    idx = int(label)

    return labels[idx]

paths, names = get_images()
name_label = {}
# output_labels()
for i , img_path in enumerate(paths):
    # output_labels()
    name = names[i]
    lbl = label_image(img_path,name, wait_time=1000)
    name_label[name] = lbl





"""
Plan:
Read in first image of each folder with folder name

Then for each image, output the image, user provides short input for label
Store this label in dict where file name maps to label
"""
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

labels = ['Accessory',
          
          'Long sleeved top',
          'Short sleeved top',
          'Coat',
          'Dress',
          'Trouser',
          'Skirt',
          'Shorts',
          
          'Bag',
          'Wallet/Purse',
          'Belt',
          'Ladies pants',
          'Briefs',
          'Ear ring',
          'Glasses',
          'Sun glasses',
          
          'Hat',
          
          'Sandal',
          'Trainer',
          'Smart shoes',
          
          'Necklace',
          'Wrist wear',
          'Unknown',]
images_path = "../../uob_image_set"


class ImageLabeler():

    def __init__(self, path, labelled_path, unlabelled_path):
        self.path = path
        self.labelled_path = labelled_path
        self.unlabelled_path = unlabelled_path
        self.labelled_folders = []
        with open(labelled_path, 'r') as f:
            labelled_folders = f.read().split("\n")
            while '' in self.labelled_folders: self.labelled_folders.remove('')

            if labelled_folders is not None:
                labelled_folders = [line.split(",") for line in labelled_folders]
                self.labelled_folders.extend(labelled_folders)

        with open(unlabelled_path, 'r') as f:
            self.unlabelled_folders = f.read().split("\n")

        while '' in self.unlabelled_folders: self.unlabelled_folders.remove('')

        self.get_images()

    def get_images(self):
        folders = os.listdir(self.path)
        print("Loading Images...")
        folders = [folder for folder in folders if folder.isdigit()]
        paths = [self.full_path(folder) for folder in tqdm(folders)]

        print("LOADED")
        self.img_paths = paths
        self.names = folders

    def full_path(self, dir_name):
        image_folder = os.path.join(self.path, dir_name)

        img_name = os.listdir(image_folder)[0]
        return os.path.join(image_folder, img_name)

    def label_image(self, idx, wait_time=0):
        # If wait time is 0, then you have to click a key to close the window
        image_path, name = self.img_paths[idx], self.names[idx]

        img = cv2.imread(image_path)
        img = cv2.resize(img, (500, 667))
        cv2.imshow(name, img)
        cv2.waitKey(wait_time)
        cv2.destroyWindow(name)
        label = input("Write number to represent label: ")
        idx = int(label)

        return labels[idx]

    def label_n_images(self, n=1):

        if n <= 0 or n > len(self.unlabelled_folders):
            n = len(self.unlabelled_folders)

        chosen_dirs = np.random.choice(self.unlabelled_folders, n)
        for img_name in chosen_dirs:
            i = self.names.index(img_name)
            lbl = self.label_image(i, wait_time=1000)

            if lbl != "Unknown":  # If something other than unknown is picked
                self.labelled_folders.append((img_name, lbl))
                self.unlabelled_folders.remove(img_name)

        self.update_files()


    def update_files(self):
        with open(self.labelled_path, "w") as f:
            lines = [",".join(line) for line in self.labelled_folders]
            text = "\n".join(lines)
            f.write(text)

        with open(self.unlabelled_path, "w") as f:
            text = "\n".join(self.unlabelled_folders)
            f.write(text)



def output_labels():
    for i, lbl in enumerate(labels):
        print(i, ".", lbl)


if __name__ == '__main__':
    while True:
        labeler = ImageLabeler(images_path, "labelled.csv", "unlabelled.txt")
        output_labels()
        labeler.label_n_images(n=10)

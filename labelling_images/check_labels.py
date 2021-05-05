import pandas as pd
import os
import cv2
from tqdm import tqdm
import numpy as np

class LabelChecker():

    def __init__(self, path, labelled_path, unlabelled_path, labels_to_check):
        self.path = path
        self.labelled_path = labelled_path
        self.unlabelled_path = unlabelled_path
        labelled_folders = read_file(labelled_path)
        self.labelled_folders = [line.split(",") for line in labelled_folders]

        self.folders_to_check = [row for row in self.labelled_folders if row[1] in labels_to_check]

        self.unlabelled_folders = read_file(unlabelled_path)

        self.checked_path = "checked.txt"
        self.checked = read_file(self.checked_path)
        self.folders_to_check = [row for row in self.folders_to_check if not (row[0] in self.checked)]


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

        img_name = [file for file in os.listdir(image_folder) if "_0" in file][0]
        return os.path.join(image_folder, img_name)

    def remove_from_labelled(self,name_label):
        self.labelled_folders.remove(name_label)
        self.unlabelled_folders.append(name_label[0])

    def add_to_checked(self,folder):
        self.checked.append(folder)

    def check_folder(self,name_label):
        name, label = name_label
        path = self.path + "/" + name + "/" + name + "_0.jpg"
        img = cv2.imread(path)
        img = cv2.resize(img, (500, 667))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow(label, img)
        cv2.waitKey(0)
        cv2.destroyWindow(label)

        print(" 1. Correct \n 2.Incorrect \n 3. No Clue")
        answer = input("Type 1 or 2: ")

        if answer == "1":
            self.add_to_checked(name)
        elif answer == "2":
            self.remove_from_labelled(name_label)

    def check_n_images(self, n=1):

        if n <= 0 or n > len(self.labelled_folders):
            n = len(self.labelled_folders)

        chosen_idxs = np.random.choice(len(self.folders_to_check), n, replace=False)

        for i in chosen_idxs:
            name_label = self.folders_to_check[i]
            self.check_folder(name_label)


        self.update_files()


    def update_files(self):
        with open(self.labelled_path, "w") as f:
            lines = [",".join(line) for line in self.labelled_folders]
            text = "\n".join(lines)
            f.write(text)

        with open(self.unlabelled_path, "w") as f:
            text = "\n".join(self.unlabelled_folders)
            f.write(text)

        with open(self.checked_path, "w") as f:
            text = "\n".join(self.checked)
            f.write(text)


def read_file(path):
    with open(path, 'r') as f:
        lines = f.read().split("\n")

    while '' in lines: lines.remove('')

    return lines



image_path = "../../uob_image_set"

if __name__ == '__main__':
    toCheck = ["Accessory", "Sandal", "Smart shoes", "Trainer"]
    checker = LabelChecker(image_path, "labelled.csv", "unlabelled.txt", toCheck)
    checker.check_n_images(n=1)

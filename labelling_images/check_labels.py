import pandas as pd
import os
import cv2


def read_file(path):
    with open(path, 'r') as f:
        lines = f.read().split("\n")

    while '' in checked: checked.remove('')

    return lines

labelled_folders = read_file("labelled.csv")
labelled_folders = [line.split(",") for line in labelled_folders]

unlabelled_folders = read_file("unlabelled.txt")

checked = read_file("checked.txt")



to_check = ["Accessory", "Sandal", "Smart shoes", "Trainer"]

shoes = [row for row in labelled_folders if row[1] in to_check]

image_path = "../../uob_image_set"

def get_image_path(folder):

    path = os.path.join(image_path, folder, folder + "_0.jpg")
    return path

def remove_from_labelled(folder):
    to_remove = [row for row in labelled_folders if row[0] != folder][0]
    labelled_folders.remove(to_remove)
    unlabelled_folders.append(folder)

def add_to_checked(folder):
    checked.append(folder)

def check_folder(folder):
    path = get_image_path(folder)
    img = cv2.imread(path)
    img = cv2.resize(img, (500, 667))

    label = [label for folder,label in labelled_folders][0]

    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyWindow(folder)

    print(" 1. Correct \n 2.Incorrect")
    answer = input("Type 1 or 2: ")

    if answer == "1":
        add_to_checked(folder)
    elif answer == "2":
        remove_from_labelled(folder)




print(checked)
from __future__ import absolute_import
import sys, os

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)
import random

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import Dataset_CNN.generate_error_matrix as g_e_m
import torch
import pickle
import pandas as pd
##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

class ClothesFolder(ImageFolder):
    
    def __init__(self,root , transform=None):
        super(ClothesFolder, self).__init__(root=root, transform = transform)
        name  = root.split("/")[-1]
        if not any([name == p for p in os.listdir("data")]):
            g_e_m.generate_matrix(root, name)

        self.labels_to_folder, self.folder_to_labels = self.convert_to_dict(pd.read_csv("../labelling_images/labelled.csv"))

        self.data_path = "data/" + name + "/"


        self.images = {}
        for dir in self.classes:
            class_index = self.class_to_idx[dir]
            self.images[class_index] = []

        for s in self.samples:
            if s[1] in self.images.keys():
                self.images[s[1]].append(s[0])

        self.classdictsize = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdictsize[ci] = len(self.images[ci])



    def load_and_transform(self, path, img_name):
        img = Image.open(path + "/" + img_name)

        if self.transform:
            img =  self.transform(img)

        return np.array(img)

    def __getitem__(self,index):
        anchor_path, anchor_target = self.samples[index]
        anchor_name = anchor_path.split("\\")[-1].split(".")[0]

        pos_paths = [i for i in self.images[anchor_target] if i != anchor_path]
        positive_error_diffs = self.load_diff_dict("positives", anchor_name)
        positives = list(positive_error_diffs.items())
        names, distances = list(zip(*positives))
        weights = self.get_probabilties(distances, exp_sign = 1 )
        pos_path = np.random.choice(pos_paths, p = weights)


        #to do next:weight closer images higher in random choice
        negatives = self.get_closest(anchor_name,5)
        names, distances = list(zip(*negatives))
        weights = self.get_probabilties(distances, exp_sign = 1 )
        neg_name = np.random.choice(names, p = weights)
        neg_idx = self.class_to_idx[neg_name.split("_")[0]]
        paths_neg_dir = self.images[neg_idx]
        neg_path = [p for p in paths_neg_dir if neg_name in p][0]
        #find our class
        #from our class, find other classes which are similar based on fft of the first image in the class
        #return a list of x classes and for each class, choose a random image in the class as a negative
        # print(pos_path)

        # Load the data for these samples.
        a_sample = self.loader(anchor_path)
        p_sample = self.loader(pos_path)
        n_sample = self.loader(neg_path)

        # apply transforms
        if self.transform is not None:
            a_sample = self.transform(a_sample)
            p_sample = self.transform(p_sample)
            n_sample = self.transform(n_sample)


        # note that we do not return the label!
        return a_sample, p_sample, n_sample

    def get_closest(self,image_name, k):
        negative_error_diffs = self.load_diff_dict("negatives", image_name)
        isLabelled = image_name[:8] in self.folder_to_labels.keys()

        row = list(negative_error_diffs.items())
        if isLabelled:
            label = self.folder_to_labels[image_name[:8]]
            other_folders = self.labels_to_folder[label]
            row = list(filter(lambda r: r[0][:8] in other_folders,row))

        if len(row) == 0: row = list(negative_error_diffs.items())
        k_smallest_idx = sorted(row,key=lambda x: x[1])[0:k]
        return k_smallest_idx

    def get_probabilties(self, distances , exp_sign = 1):
        exponetial_list = []
        for i in distances:
            exponetial = np.exp(exp_sign * i)
            exponetial_list.append(exponetial)

        batch_weighted_triplet_list = []
        sum_exp = sum(exponetial_list)
        for e in exponetial_list:
            weighted_triplet = e / sum_exp
            batch_weighted_triplet_list.append(weighted_triplet)
        return batch_weighted_triplet_list

    def load_diff_dict(self, folder, image_name):
        path = os.path.join(self.data_path, folder, image_name + ".pickle")
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            return pickle.load(f)


    def test_output_k_closest(self, idx, k = 5):

        anchor_path, anchor_target = self.samples[idx]

        anchor_class = anchor_path.split("\\")[-1].split(".")[0]
        # to do next:weight closer images higher in random choice
        neg_names = self.get_closest(anchor_class, k)
        neg_paths = []
        for name,_ in neg_names:
            neg_idx = self.class_to_idx[name.split("_")[0]]
            paths_neg_dir = self.images[neg_idx]
            neg_path = [p for p in paths_neg_dir if name in p][0]
            neg_paths.append(neg_path)


        # Load the data for these samples.
        a_sample = self.loader(anchor_path)
        neg_samples = []
        for path in neg_paths:
            n_sample = self.loader(path)
            neg_samples.append(n_sample)

        # apply transforms
        if self.transform is not None:
            a_sample = self.transform(a_sample)
            for i,img in enumerate(neg_samples):
                neg_samples[i] = self.transform(img)

        # note that we do not return the label!
        return [a_sample] +  neg_samples

    def convert_to_dict(self, labelled_df):
        label_to_folder = {}
        folder_to_label = {}
        for index, row in labelled_df.iterrows():
            label, folder = row["label"], str(row["folder"])
            if not (label in label_to_folder.keys()):
                label_to_folder[label] = []

            label_to_folder[label].append(folder)
            folder_to_label[folder] = label

        return label_to_folder, folder_to_label


# images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'

def show_example_triplet(triple):
    anchor_im, positive_im, negative_im = triple

    dst = Image.new('RGB', (anchor_im.width + positive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(positive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + positive_im.width, 0))
    dst.show()

def getListImages(images):
    h,w = images[0].height, images[0].width
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x,y))
        x+= w

    return dst

def showImages(net, old):
    net_images = getListImages(net)
    # old_images = getListImages(old)
    #
    # h,w = net_images.height , net_images.width
    # dst = Image.new('RGB', ( w, h * 2))
    # x = 0
    # y = 0
    # for i in [net_images]:
    #     dst.paste(i, (x, y))
    #     y += h

    net_images.show()




if __name__ == '__main__':
    images_path = "../../uob_image_set"

    transform = transforms.Resize((1333//5,1000//5))
    # dataset = old.ClothesFolder(images_path, transform = transform)
    dataset_net = ClothesFolder(images_path, transform = transform)


    for i in range(5):
        i = random.randint(0, 1500)
        print(i)
        test_net = dataset_net[i]
        # test_old = dataset.output_k_closest(i,10)
        showImages(test_net, [])
    # print(test)


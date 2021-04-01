import pickle

import cv2
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from PIL import Image, ImageStat
import torch
import numpy as np
import os
import random
from matplotlib import cm
from torchvision.datasets import ImageFolder
from tqdm import tqdm

##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973


class ClothesDataset(Dataset):

    def __init__(self,image_path, n , transform=None, sample_for_negatives = 100, load_data = False):
        self.root = image_path
        self.max_images = n
        self.transform = transform
        self.samples = sample_for_negatives

        if load_data:

            image_directory = os.listdir(self.root)
            image_directory = image_directory if n == 0 or n >= len(image_directory) else image_directory[:n]
            images = {}
            i = 0

            for dir in tqdm(image_directory):
                path = os.path.join(self.root, dir)
                images[dir] = [self.load_and_transform(path,img_name) for img_name in os.listdir(path)]
                i += len(images[dir])

            self.images = images
            self.total_imgs = i
            with open('images.p', 'wb') as fp:
                pickle.dump(images, fp, protocol=pickle.HIGHEST_PROTOCOL)

            print("Loaded")

        else:
            with open('images.p', 'rb') as fp:
                images = pickle.load(fp)
                self.total_imgs = sum(list(map(len, images.values())))
                self.images = images
                print("Loaded")

    def __len__(self):
        return self.total_imgs

    def load_and_transform(self, path, img_name):
        img = Image.open(path + "/" + img_name)

        if self.transform:
            img =  self.transform(img)

        return np.array(img)

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.images.keys()))
        chosen_dir,dir_imgs = list(self.images.items())[idx]

        idxs = np.arange(1, len(dir_imgs))
        a_idx, p_idx = 0, np.random.choice(idxs)
        anchor, positive = dir_imgs[a_idx], dir_imgs[p_idx]

        other_dirs = list(self.images.keys())
        other_dirs.remove(chosen_dir)
        negative = self.get_close_negative(anchor, other_dirs)

        triple = [anchor,positive,negative]

        return tuple(triple)

    def get_close_negative(self, anchor, other_dirs):
        w, h, _ = anchor.shape
        anchor = Image.fromarray(anchor)

        negative_images_dir = random.sample(other_dirs, self.samples)
        smallest_diff = None

        for n_dir in tqdm(negative_images_dir):
            neg_img = Image.fromarray(self.images[n_dir][0])

            neg_diff = 1/(w * h) * np.sum(abs(anchor - neg_img))

            if smallest_diff == None or smallest_diff > neg_diff:
                smallest_diff = neg_diff
                closest_negative = neg_img

        return closest_negative

class ClothesFolder(ImageFolder):
    
    def __init__(self,image_path, n , transform=None, sample_for_negatives = 100, load_data = False):
        super(ClothesFolder, self).__init__(root=image_path, transform = transform)
        # super().__init__(root=image_path, transform = transform,sample_for_negatives,load_data)
    
        self.max_images = n
        # self.samples = sample_for_negatives
        self.num_samples = 100
        self.col_diff = np.load('col_diff.npy',allow_pickle=True).item()
        self.fft_diff = np.load('fft_diff.npy',allow_pickle=True).item()
        # print(self.fft_diff.keys())
        # print(self.classes)

        # if load_data:
        #     # Create a dictionary of lists for each class for reverse lookup
        #     # to generate triplets 
        #     images= {}
        #     for c in self.classes:
        #         ci = self.class_to_idx[c]
        #         images[ci] = []
        #     # append each file in the approach dictionary element list
        #     for s in self.samples:
        #         images[s[1]].append(s[0])
        #     # images[s[1]] = [self.load_and_transform(path,img_name) for img_name in os.listdir(path)]
        #     # keep track of the sizes for random sampling
        #     # self.total_imgs = {}
        #     # for c in self.classes:
        #     #     ci = self.class_to_idx[c]
        #     #     self.total_imgs[ci] = len(images[ci])
        #     self.total_imgs = sum(list(map(len, images.values())))
        #     self.images = images
        #     with open('images.p', 'wb') as fp:
        #         pickle.dump(images, fp, protocol=pickle.HIGHEST_PROTOCOL)
        self.images = {}
        if load_data:
            for dir in self.classes:
                class_index = self.class_to_idx[dir]
                self.images[class_index] = []
            for s in self.samples:
                self.images[s[1]].append(s[0])
            self.classdictsize = {}
            for c in self.classes:
                ci = self.class_to_idx[c]
                self.classdictsize[ci] = len(self.images[ci])    

            # print(self.samples[0])
        else:
            with open('images.p', 'rb') as fp:
                self.images = pickle.load(fp)
                self.total_imgs = sum(list(map(len, self.images.values())))
                # self.total_imgs = {}
                # for c in self.classes:
                #     ci = self.class_to_idx[c]
                #     self.total_imgs[ci] = len(images[ci])
                print("Loaded")     
    
    # def __len__(self):
    #     return self.classdictsize

    def load_and_transform(self, path, img_name):
        img = Image.open(path + "/" + img_name)

        if self.transform:
            img =  self.transform(img)

        return np.array(img)

    def __getitem__(self,index):
        anchor_path, anchor_target = self.samples[index]
        pos_images = [i for i in self.images[anchor_target] if i != anchor_path]
        pos_path = random.choice(pos_images)
        anchor_class = self.classes[anchor_target]
        #to do next:weight closer images higher in random choice 
        neg_dir = random.choice(self.get_closest(anchor_class,3))[0]
        neg_images = [i for i in self.images[self.class_to_idx[neg_dir]] if i != anchor_path]
        neg_path = random.choice(neg_images)
        print(anchor_path)
        print(pos_path)
        print(neg_path)
        #find our class
        #from our class, find other classes which are similar based on fft of the first image in the class
        #return a list of x classes and for each class, choose a random image in the class as a negative
        # print(pos_path)
        return tuple([anchor_path,pos_path,neg_path])

    def get_closest(self,class_name, k):
        images = np.array(self.images)
        row = list(self.fft_diff[class_name].items())
        k_smallest_idx = sorted(row,key=lambda x: x[1])[1:k+1]
        return k_smallest_idx

    def get_close_negative(self, anchor, other_dirs):
        # print(anchor)
        anchor = Image.open(anchor)
        # w, h, _ = anchor.shape
        w,h = anchor.width,anchor.height
        # anchor = Image.fromarray(anchor)

        negative_images_dir = random.sample(other_dirs, self.samples)
        smallest_diff = None

        for n_dir in tqdm(negative_images_dir):
            neg_img = Image.fromarray(self.images[n_dir][0])

            neg_diff = 1/(w * h) * np.sum(abs(anchor - neg_img))

            if smallest_diff == None or smallest_diff > neg_diff:
                smallest_diff = neg_diff
                closest_negative = neg_img

        return closest_negative    
                 
# images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'
images_path = "../../uob_image_set"
single_view_path = "../../uob_image_set_0"



if __name__ == '__main__':

    transform = transforms.Resize((1333,1000))
    # dataset = ClothesDataset(images_path, 500, transform = transform, sample_for_negatives= 99, load_data=False)
    dataset = ClothesFolder(images_path, 500, transform = transform, sample_for_negatives= 100, load_data=True)
    dataloader = DataLoader(dataset, batch_size = 5, shuffle=True)
    i = random.randint(0,1500)
    test = dataset[i]
    # print(test)
    anchor , positive , negative = test

    # anchor_im = Image.fromarray(np.uint8(anchor))
    # postive_im = Image.fromarray(np.uint8(positive))
    # negative_im = Image.fromarray(np.uint8(negative))
    anchor_im = Image.open(anchor)
    positive_im = Image.open(positive)
    negative_im = Image.open(negative)

    dst = Image.new('RGB', (anchor_im.width + positive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(positive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + positive_im.width, 0))
    dst.show()


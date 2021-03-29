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



images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'



if __name__ == '__main__':

    transform = transforms.Resize((1333,1000))
    dataset = ClothesDataset(images_path, 500, transform = transform, sample_for_negatives= 99, load_data=False)
    dataloader = DataLoader(dataset, batch_size = 5, shuffle=True)

    test = dataset[20]
    anchor , positive , negative = test

    anchor_im = Image.fromarray(np.uint8(anchor))
    postive_im = Image.fromarray(np.uint8(positive))
    negative_im = Image.fromarray(np.uint8(negative))

    dst = Image.new('RGB', (anchor_im.width + postive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(postive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + postive_im.width, 0))
    dst.show()

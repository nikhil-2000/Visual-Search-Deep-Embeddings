import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from PIL import Image
import torch
import numpy as np
import os
import random
from matplotlib import cm
##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973


class ClothesDataset(Dataset):

    def __init__(self,image_path, n , transform=None, sample_for_negatives = 100):

        self.root = image_path
        self.max_images = n
        self.transform = transform
        self.samples = sample_for_negatives

        image_directory = os.listdir(self.root)
        image_directory = image_directory if n == 0 or n >= len(image_directory) else image_directory[:n]
        images = {}
        i = 0

        for dir in image_directory:
            path = os.path.join(self.root, dir)
            images[dir] = [path + "/" + img_name for img_name in os.listdir(path)]
            i += len(images[dir])

        self.images = images
        self.total_imgs = i


    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.images.keys()))
        chosen_dir,dir_imgs = list(self.images.items())[idx]


        idxs = np.arange(0, len(dir_imgs))
        a_idx, p_idx = np.random.choice(idxs, size=2, replace=False)
        anchor, positive = dir_imgs[a_idx], dir_imgs[p_idx]

        other_dirs = list(self.images.keys())
        other_dirs.remove(chosen_dir)
        negative = self.get_close_negative(anchor, other_dirs)

        img_names = [anchor,positive,negative]
        triple = []

        for name in img_names:
            img = Image.open(name)

            if self.transform:
                img = self.transform(img)

            triple.append(np.array(img))




        return tuple(triple)

    def get_close_negative(self, anchor, other_dirs):

        anchor = Image.open(anchor)
        if self.transform:
            anchor = self.transform(anchor)

        anchor = np.array(anchor)

        negative_images_dir = random.sample(other_dirs, self.samples)
        smallest_diff = None

        for i,n_dir in enumerate(negative_images_dir):
            neg_img_paths = self.images[n_dir]

            for i,neg_img_path in enumerate(neg_img_paths):
                neg_img = Image.open(neg_img_path)

                if self.transform:
                    neg_img = self.transform(neg_img)

                neg_img = np.array(neg_img)

                diff = np.sum((anchor.astype("float") - neg_img.astype("float")) ** 2)

                if (smallest_diff == None):
                    smallest_diff = diff

                if smallest_diff > diff:
                    smallest_diff = diff
                    closest_negative_folder = n_dir
                    closest_idx = i


        # print(diff)
        return self.images[closest_negative_folder][closest_idx]








images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'



if __name__ == '__main__':

    transform = transforms.Resize((1333,1000))
    dataset = ClothesDataset(images_path, 1000, transform = transform, sample_for_negatives= 100)
    dataloader = DataLoader(dataset, batch_size = 5, shuffle=True)

    test = dataset[10]
    anchor , positive , negative = test

    anchor_im = Image.fromarray(np.uint8(anchor))
    postive_im = Image.fromarray(np.uint8(positive))
    negative_im = Image.fromarray(np.uint8(negative))

    dst = Image.new('RGB', (anchor_im.width + postive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(postive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + postive_im.width, 0))
    dst.show()


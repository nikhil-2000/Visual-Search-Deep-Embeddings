import os
import numpy as np
import torch
from PIL import Image

images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'

##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973


def load_images_by_dir(n = 0):

    image_directory = os.listdir(images_path)
    image_directory = image_directory if n == 0 or n >= len(image_directory) else image_directory[:n]
    images = {}
    i = 0

    for dir in image_directory:
        path = os.path.join(images_path,dir)
        dir_imgs = [Image.open(path + "/" + img_name) for img_name in os.listdir(path)]

        images[dir] = [np.array(img) for img in dir_imgs]
        i+= 1

        if i % 100 == 0:
            print(i , "/ ", n)


    return images

def create_triples(images):
    triples = []
    i = 0
    for dir,dir_imgs in images.items():
        idxs = np.arange(0,len(dir_imgs))
        a_idx,p_idx = np.random.choice(idxs,size = 2, replace = False)
        anchor, positive = dir_imgs[a_idx], dir_imgs[p_idx]

        other_dirs = list(images.keys())
        other_dirs.remove(dir)
        negative_dir = np.random.choice(other_dirs)
        negative = images[negative_dir][0]

        triples.append(np.array([anchor,positive,negative]))

    triples = np.array(triples)
    print(triples.shape)
    # np.save("triples.npy", triples)
    return triples

# n = 1500 all images
n = 1500

imgs = load_images_by_dir(n)
triples = create_triples(imgs)
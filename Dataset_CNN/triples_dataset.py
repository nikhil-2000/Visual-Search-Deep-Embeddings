import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import triplet_dataset_old as old


##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

class ClothesFolder(ImageFolder):
    
    def __init__(self,root , transform=None):
        super(ClothesFolder, self).__init__(root=root, transform = transform)
        # super().__init__(root=image_path, transform = transform,sample_for_negatives,load_data)

        self.error_diff = np.load("../error_net_diff.npy", allow_pickle=True).item()

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
        pos_images = [i for i in self.images[anchor_target] if i != anchor_path]
        pos_path = random.choice(pos_images)
        anchor_class = anchor_path.split("\\")[-1].split(".")[0]
        #to do next:weight closer images higher in random choice 
        neg_name = random.choice(self.get_closest(anchor_class,5))[0]
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

    def get_closest(self,class_name, k):
            row = list(self.error_diff[class_name].items())
            k_smallest_idx = sorted(row,key=lambda x: x[1])[1:k+1]
            return k_smallest_idx

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
                 
# images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'
images_path = "../../uob_image_set_100"
single_view_path = "../../uob_image_set_0"

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
    old_images = getListImages(old)

    h,w = net_images.height , net_images.width
    dst = Image.new('RGB', ( w, h * 2))
    x = 0
    y = 0
    for i in [net_images, old_images]:
        dst.paste(i, (x, y))
        y += h

    dst.show()



if __name__ == '__main__':

    transform = transforms.Resize((1333//5,1000//5))
    dataset = old.ClothesFolder(images_path, transform = transform)
    dataset_net = ClothesFolder(images_path, transform = transform)


    for i in range(5):
        i = random.randint(0, 100)
        print(i)
        test_net = dataset_net.test_output_k_closest(i , 10)
        test_old = dataset.output_k_closest(i,10)
        showImages(test_net, test_old)
    # print(test)


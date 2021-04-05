import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

class ClothesFolder(ImageFolder):
    
    def __init__(self,image_path, n , transform=None):
        super(ClothesFolder, self).__init__(root=image_path, transform = transform)
        # super().__init__(root=image_path, transform = transform,sample_for_negatives,load_data)
    
        self.max_images = n

        self.error_diff = np.load('error_diff.npy',allow_pickle=True).item()

        self.images = {}
        for dir in self.classes:
            class_index = self.class_to_idx[dir]
            self.images[class_index] = []
        for s in self.samples:
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
        anchor_class = self.classes[anchor_target]
        #to do next:weight closer images higher in random choice 
        neg_dir = random.choice(self.get_closest(anchor_class,1))[0]
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
        row = list(self.error_diff[class_name].items())
        k_smallest_idx = sorted(row,key=lambda x: x[1])[1:k+1]
        return k_smallest_idx

                 
# images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'
images_path = "../../uob_image_set"
single_view_path = "../../uob_image_set_0"



if __name__ == '__main__':

    transform = transforms.Resize((1333,1000))
    # dataset = ClothesDataset(images_path, 500, transform = transform, sample_for_negatives= 99, load_data=False)
    dataset = ClothesFolder(images_path, 500, transform = transform)
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


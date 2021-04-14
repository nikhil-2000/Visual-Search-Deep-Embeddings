import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

class ClothesFolder(ImageFolder):

    def __init__(self, root, transform=None):
        super(ClothesFolder, self).__init__(root=root, transform=transform)
        # super().__init__(root=image_path, transform = transform,sample_for_negatives,load_data)

        self.error_diff = np.load('../Nikhil/error_diff.npy', allow_pickle=True).item()

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
            img = self.transform(img)

        return np.array(img)

    def __getitem__(self, index):
        anchor_path, anchor_target = self.samples[index]
        pos_images = [i for i in self.images[anchor_target] if i != anchor_path]
        pos_path = random.choice(pos_images)
        anchor_class = self.classes[anchor_target]
        # to do next:weight closer images higher in random choice
        neg_dir = random.choice(self.get_closest(anchor_class, 3))[0]
        neg_images = [i for i in self.images[self.class_to_idx[neg_dir]] if i != anchor_path]
        neg_path = random.choice(neg_images)
        # find our class
        # from our class, find other classes which are similar based on fft of the first image in the class
        # return a list of x classes and for each class, choose a random image in the class as a negative
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

    def get_closest(self, class_name, k):
        row = list(self.error_diff[class_name].items())
        k_smallest_idx = sorted(row, key=lambda x: x[1])[1:k + 1]
        return k_smallest_idx

    def output_k_closest(self, idx, k = 5):

        anchor_path, anchor_target = self.samples[idx]
        pos_images = [i for i in self.images[anchor_target] if i != anchor_path]
        pos_path = random.choice(pos_images)
        anchor_class = self.classes[anchor_target]
        # to do next:weight closer images higher in random choice
        neg_dirs = self.get_closest(anchor_class, k)
        neg_paths = []
        for name,_ in neg_dirs:
            neg_images = [i for i in self.images[self.class_to_idx[name]] if i != anchor_path]
            neg_path = random.choice(neg_images)
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
images_path = "../../uob_image_set"

if __name__ == '__main__':
    transform = transforms.Resize((1333, 1000))
    # dataset = ClothesDataset(images_path, 500, transform = transform, sample_for_negatives= 99, load_data=False)
    dataset = ClothesFolder(images_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    i = random.randint(0, 1500)
    test = dataset.output_k_closest(5)
    # print(test)
    # anchor_im, positive_im, negative_im = test
    #
    # dst = Image.new('RGB', (anchor_im.width + positive_im.width + negative_im.width, anchor_im.height))
    # dst.paste(anchor_im, (0, 0))
    # dst.paste(positive_im, (anchor_im.width, 0))
    # dst.paste(negative_im, (anchor_im.width + positive_im.width, 0))
    # dst.show()

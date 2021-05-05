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
from Dataset_CNN.EmbeddingNetwork import EmbeddingNetwork
import matplotlib.pyplot as plt
from tqdm import tqdm

##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 100
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ClothesFolder(ImageFolder):

    def __init__(self, root, transform=None, margin = 1.0):
        super(ClothesFolder, self).__init__(root=root, transform=transform)
        name = os.path.basename(os.path.normpath(root))
        # if not any([name == p for p in os.listdir("data")]):
        #     g_e_m.generate_matrix(root, name)

        self.labels_to_folder, self.folder_to_labels = self.convert_to_dict(
            pd.read_csv("../labelling_images/labelled.csv"))

        self.data_path = "data/" + name + "/"
        self.remaining_folders = list(self.folder_to_labels.keys())

        self.folder_to_batch = {}
        self.batches = []
        self.batch_distances = []

        self.training_labels = True

        self.modelfile = None

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

        self.margin = margin

    def load_model(self):

        model = EmbeddingNetwork()
        if self.modelfile is not None:
            checkpoint = torch.load(self.modelfile)
            model.load_state_dict(checkpoint['model_state_dict'])
            # model = torch.jit.script(model).to(device) # send model to GPU
        return model

    def __getitem__(self, index):
        #Grabs the sample and batch
        anchor_path, anchor_target = self.samples[index]
        anchor_name = anchor_path.split("\\")[-1].split(".")[0]
        folder_name = anchor_name.split("_")[0]
        batch_idx = self.folder_to_batch[folder_name]

        #Gets all paths + distances in same folder
        if not self.training_labels:
            pos_paths = [i for i in self.images[anchor_target] if i != anchor_path]
            positive_error_diffs = self.batch_distances[batch_idx][1][anchor_name + ".jpg"]

            #Weights choices (Not used rn)
            positives = list(positive_error_diffs.items())
            names, pos_distances = list(zip(*positives))
            weights = self.get_probabilties(pos_distances, exp_sign=1)
            # pos_path = np.random.choice(pos_paths, p=weights)

            #Chooses the furthest positives all the time
            pos_path = pos_paths[-1]

        elif self.training_labels:
            #Grabs all folders of same label in batch
            anchor_label = self.folder_to_labels[folder_name]

            #Grabs all "positives" with same label
            other_folder_distances = list(self.batch_distances[batch_idx][0][anchor_name + ".jpg"].items())
            same_folder_distances = list(self.batch_distances[batch_idx][1][anchor_name + ".jpg"].items())
            other_folder_distances.extend(same_folder_distances)
            positives = [(name,dist) for (name,dist) in other_folder_distances if self.folder_to_labels[name.split("_")[0]] == anchor_label]

            #Picks a positive based on distances, weights further postives higher
            names, pos_distances = list(zip(*positives))
            weights = self.get_probabilties(pos_distances, exp_sign=1)
            pos_name = np.random.choice(names, p=weights)

            pos_idx = self.class_to_idx[pos_name.split("_")[0]]
            paths_pos_dir = self.images[pos_idx]
            pos_path = [p for p in paths_pos_dir if pos_name in p][0]


        # to do next:weight closer images higher in random choice
        # negatives = self.get_closest_v2(anchor_name, 5)

        # names, distances = list(zip(*negatives))
        # weights = self.get_probabilties(distances, exp_sign=1)
        # neg_name = np.random.choice(names, p=weights)

        #Gets a semihard negative, explained in function
        neg_name,neg_distance = self.get_semi_hard_negative(anchor_name, pos_distances[-1])
        neg_idx = self.class_to_idx[neg_name.split("_")[0]]
        paths_neg_dir = self.images[neg_idx]
        neg_path = [p for p in paths_neg_dir if neg_name in p][0]
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


    # def get_closest(self, image_name, k):
    #     negative_error_diffs = self.load_diff_dict("negatives", image_name)
    #     isLabelled = image_name[:8] in self.folder_to_labels.keys()
    #
    #     row = list(negative_error_diffs.items())
    #     if isLabelled:
    #         label = self.folder_to_labels[image_name[:8]]
    #         other_folders = self.labels_to_folder[label]
    #         row = list(filter(lambda r: r[0][:8] in other_folders, row))
    #
    #     if len(row) == 0: row = list(negative_error_diffs.items())
    #     k_smallest_idx = sorted(row, key=lambda x: x[1])[0:k]
    #     return k_smallest_idx
    #
    # def get_closest_v2(self, image_name, k):
    #     folder_name = image_name.split("_")[0]
    #     batch_idx = self.folder_to_batch[folder_name]
    #     batch_distances = self.batch_distances[batch_idx][0][image_name + ".jpg"]
    #
    #     isLabelled = folder_name in self.folder_to_labels.keys()
    #
    #     row = list(batch_distances.items())
    #     if len(row) == 0:
    #         folder_names = random.choices(list(self.class_to_idx.keys()), k=k)
    #         return [(name, 1) for name in folder_names]
    #
    #     if isLabelled:
    #         label = self.folder_to_labels[image_name[:8]]
    #         other_folders = self.labels_to_folder[label]
    #         row = list(filter(lambda r: r[0][:8] in other_folders, row))
    #
    #     if len(row) == 0: row = list(batch_distances.items())
    #     k_closest = sorted(row, key=lambda x: x[1])[0:k]
    #     return k_closest

    def get_semi_hard_negative(self,image_name,pos_dist):
        #Getting the batch distances
        folder_name = image_name.split("_")[0]
        anchor_label = self.folder_to_labels[folder_name]
        batch_idx = self.folder_to_batch[folder_name]
        batch_distances = self.batch_distances[batch_idx][0][image_name + ".jpg"]

        row = list(batch_distances.items())

        #Select a negative of different label
        if self.training_labels:
            label = self.folder_to_labels[image_name[:8]]
            other_folders = self.labels_to_folder[label]
            row = [(name,dist) for (name,dist) in row if self.folder_to_labels[name.split("_")[0]] != anchor_label]

        #Select a negative of a same label
        elif not self.training_labels:
            label = self.folder_to_labels[image_name[:8]]
            other_folders = self.labels_to_folder[label]
            row = [(name,dist) for (name,dist) in row if self.folder_to_labels[name.split("_")[0]] == anchor_label]

        #Reset if there isn't any of same label in batch
        if len(row) == 0: row = list(batch_distances.items())

        #Semi-hard triplet selection, further than the positive, closer than the positive + margin
        row = filter(lambda name_dist:  pos_dist + self.margin > name_dist[1] > pos_dist , row)
        row = list(row)

        #Select triplet further than postive if no semi-hard triplets found
        if len(row) == 0:
            row = list(batch_distances.items())
            row = filter(lambda name_dist:  name_dist[1] > pos_dist , row)
            row = list(row)

        # If there is no triplets further than positive, just pick any in batch
        # May need this when the batch size is really small
        if len(row) == 0:
            row = list(batch_distances.items())

        #Currently returns random, need to implement weights
        return random.choice(row)


    def pick_batches(self, size):
        #Load all folder names that we are training on
        all_folders = list(self.class_to_idx.keys())
        n = len(all_folders)

        #Set up batches storage
        self.batches = [[] for i in range(n // size + 1)]

        #Rearranges folders randomly
        random.shuffle(all_folders)

        #Splits folders into batches using size var
        idx = 0
        for i in range(0, n, size):
            #Selects size folders and adds to batch
            batch = all_folders[i: i + size]

            #Maps folder to batch, useful when getting batch for image
            for folder in batch:
                self.folder_to_batch[folder] = idx

            self.batches[idx] = batch
            idx += 1

        while not self.batches[-1]: self.batches.remove([])

    def calc_distances(self):
        #Loads current model
        model = self.load_model()

        #Sends to GPU/CPU
        model = model.to(device)
        model.eval()
        self.batch_distances = [({}, {}) for x in range(len(self.batches))]
        #For each batch:
        # Calculate embeddings of all images
        # Get the differences between images in a folder + images with everything else
        #
        for i, batch in enumerate(tqdm(self.batches, leave=True, position=0, desc="Batch Distances")):
            embeddings, image_names = self.feed_batch(batch, model)
            neg_diff, pos_diff, _, _ = g_e_m.get_diff_dicts(embeddings, image_names)
            # Neg diff is the distances between an image in Folder A and all images in Folder B,C,D...
            # Pos diff is the dsitances between an image in Folder A and all other images in Folder A
            self.batch_distances[i] = (neg_diff, pos_diff)

    def feed_batch(self, batch, model):
        embeddings = []
        image_names = []
        #Iterate through batch
        for folder in batch:
            #Get the path to folder
            folder_path = os.path.join(self.root, folder)
            #Load each image in folder then generate embeddings
            for name in os.listdir(folder_path):
                image_names.append(name)
                image_path = os.path.join(folder_path, name)
                img = self.loader(image_path)
                if self.transform is not None:
                    img = self.transform(img)
                    c, w, h = img.shape
                    img = torch.reshape(img, (1, c, w, h))

                with torch.no_grad():
                    output = model(img.to(device))
                    cpu_tensor = output.detach().cpu()
                    pos = cpu_tensor[0].tolist()
                    embeddings.append(pos)

        embeddings = np.asarray(embeddings)

        #Return embeddings + image names to go with it
        return embeddings, image_names

    def get_probabilties(self, distances, exp_sign=1):
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


    def convert_to_dict(self, labelled_df):
        #Creates the lookups for label -> folder and folder -> label
        label_to_folder = {}
        folder_to_label = {}
        for index, row in labelled_df.iterrows():
            label, folder = row["label"], str(row["folder"])
            if not (label in label_to_folder.keys()):
                label_to_folder[label] = []

            label_to_folder[label].append(folder)
            folder_to_label[folder] = label

        return label_to_folder, folder_to_label

    def calculate_error_averages(self):
        #Calculates errors so we can track how the network is performing
        positive_losses = []
        negative_losses = []
        #Calculates a loss for each batch
        for batch in self.batch_distances:
            negative_distances_batch, positive_distances_batch = batch
            pos_loss = calc_loss(positive_distances_batch)
            neg_loss = calc_loss(negative_distances_batch, k = 50)
            positive_losses.append(pos_loss)
            negative_losses.append(neg_loss)

        return np.mean(positive_losses)/np.std(positive_losses), np.mean(negative_losses)/np.std(negative_losses)



# images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'
def calc_loss(diff_dict, k = None):
    #Grab all the losses in a batch diff dict
    losses = [list(dict_loss.values()) for dict_loss in diff_dict.values()]
    #Take k closest errors for each image in batch
    if k is not None:
        for i,loss in enumerate(losses):
            sorted_loss = sorted(loss)
            losses[i] = sorted_loss[:10]

    #Take average of errors and send back the mean
    batch_losses = []
    for loss in losses:
        avg = np.mean(loss)
        batch_losses.append(avg)

    #Represents the mean error of a batch
    return np.mean(batch_losses)



def show_example_triplet(triple):
    anchor_im, positive_im, negative_im = triple

    dst = Image.new('RGB', (anchor_im.width + positive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(positive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + positive_im.width, 0))
    dst.show()


def getListImages(images):
    h, w = images[0].height, images[0].width
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x, y))
        x += w

    return dst


def showImages(images, size=(100, 100)):
    h, w = size
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0

    for i in images:
        i = i.resize(size)
        dst.paste(i, (x, y))
        x += w

    dst.show()


def tensor_to_image(tensor):
    tensor = tensor * 255
    c, w, h = tensor.shape
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor, 'RGB')


if __name__ == '__main__':
    images_path = "../../uob_image_set_10"

    transform = transforms.Resize((1333 // 5, 1000 // 5))
    # dataset = old.ClothesFolder(images_path, transform = transform)
    dataset_net = ClothesFolder(images_path, transform=data_transforms["val"])
    dataset_net.pick_batches(2)
    dataset_net.calc_distances()
    fig, axs = plt.subplots(1, 3)
    for i in range(1):
        i = random.randint(0, 10)
        print(i)
        anchor, pos, neg = dataset_net[i]
        pil_triplet = []
        for img in [anchor, pos, neg]:
            img = img.permute(1, 2, 0)
            pil_img = tensor_to_image(img)
            pil_triplet.append(pil_img)

        pil_img.show()
        # test_old = dataset.output_k_closest(i,10)
        showImages(pil_triplet, size=(100, 100))
    # print(test)

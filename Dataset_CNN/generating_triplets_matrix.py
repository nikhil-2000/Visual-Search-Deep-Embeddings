import numpy as np
import pandas as pd
import os
import torch
import torchvision
import cv2
import pandas
import random
import faiss
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import axes3d
from  PIL import Image
from time import time
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchsummary import summary

from sklearn.decomposition import PCA
from sklearn import preprocessing

torch.cuda.is_available()

"""
Load the images
Get the embeddings for each image
all_diffs
For each image:
    Create a dictionary with {other_img_name -> distance from image}
    Add each of these dictionaries after filling in to a dictionary alldiff[image_name] = the object above
"""

resnet= models.resnet152(pretrained=True)
# resnet = models.resnet152(pretrained=True)
# freeze all base layers; Parameters of newly constructed modules have requires_grad=True by default
for param in resnet.parameters():
    param.requires_grad = False
num_ftrs = resnet.fc.in_features

import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch.tensor as tensor
n = 500
first_n = lambda x: x[:n]


embeddings = first_n(np.loadtxt("triplet_scores.txt"))
with open('triplet_files.txt') as f:
    files = f.read().split("\n")

with open('triplet_labels.txt') as f:
    labels = f.read().split("\n")

labels = [eval(lbl) for lbl in labels[:-1]]

files = first_n(files)
labels = first_n(labels)
num_img_list = [0 for i in labels]
for lbl in labels:
    num_img_list[int(lbl)] += 1

num_img_list = num_img_list[:len(labels)]
labels = ['PC' +  str(x) for x in range(1,len(num_img_list)+1)]



scaled_data = preprocessing.scale(embeddings)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

pca_df = pd.DataFrame(pca_data, columns=labels)
fig = plt.figure()
ax = plt.axes(projection="3d")


img_counter = 0
for n in num_img_list:
    r = lambda: random.randint(0,255)
    random_color = '#%02X%02X%02X' % (r(),r(),r())
    plt.plot(pca_df.PC1[img_counter:img_counter + n], pca_df.PC2[img_counter:img_counter + n],pca_df.PC3[img_counter:img_counter + n] ,'o',color=random_color)
    # plt.plot(pca_df.PC1[img_counter:img_counter + n], pca_df.PC2[img_counter:img_counter + n],'o',color=random_color)
    img_counter += n

plt.show()

"""
For each folder:
    pick random pic
    let n be number of pics in folder
    Get n - nearest pics
    
"""
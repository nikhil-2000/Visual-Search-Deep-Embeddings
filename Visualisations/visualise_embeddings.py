
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch.tensor as tensor
from tqdm import tqdm
from PIL import Image

def load_data(outfile):

    embeddings = np.loadtxt("../Dataset_CNN/data/" + outfile + "_scores.txt")
    with open('../Dataset_CNN/data/' + outfile + '_files.txt') as f:
        files = f.read().split("\n")
        if "" in files: files.remove("")

    with open('../Dataset_CNN/data/' + outfile + '_labels.txt') as f:
        labels = f.read().split("\n")
        if "" in labels: labels.remove("")


    labels = [eval(lbl) for lbl in labels]

    # files = first_n(files)
    # labels = first_n(labels)

    return embeddings, labels, files


def PCA_Output(embeddings, labels):
    num_img_list = [0 for i in labels]
    for lbl in labels:
        num_img_list[int(lbl)] += 1

    num_img_list = num_img_list[:len(labels)]
    labels = ['PC' +  str(x) for x in range(1,len(num_img_list)+2)]



    scaled_data = preprocessing.scale(embeddings)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=labels)
    fig = plt.figure()
    # ax = plt.axes(projection="3d")


    img_counter = 0
    for n in num_img_list:
        r = lambda: random.randint(0,255)
        random_color = '#%02X%02X%02X' % (r(),r(),r())
        # plt.plot(pca_df.PC1[img_counter:img_counter + n], pca_df.PC2[img_counter:img_counter + n],pca_df.PC3[img_counter:img_counter + n] ,'o',color=random_color)
        plt.plot(pca_df.PC1[img_counter:img_counter + n], pca_df.PC2[img_counter:img_counter + n],'o',color=random_color)
        img_counter += n

    plt.show()

"""
Random Example
    pick random pic
    let n be number of pics in folder
    Get n - nearest pics
    
"""

def dist_dict(scores, files, outfile):
    s = scores.shape
    dist_dict = { k : {} for k in files}

    for i,f1 in tqdm(enumerate(files)):
        for j,f2 in enumerate(files):
            if i != j:
                f1_embedding = scores[i]
                f2_embedding = scores[j]

                dist = np.sum(abs(f1_embedding - f2_embedding))

                dist_dict[f1][f2] = dist_dict[f2][f1] = dist

    np.save("../Dataset_CNN/data/" + outfile + "_diffs.npy", dist_dict)
    return dist_dict

def get_accuracy(file, dist_dict, k = 5):
    name = file.split("\\")[-2]
    row = list(dist_dict[file].items())
    k_smallest_diffs = sorted(row, key=lambda x: x[1])[0:k]
    image_paths =  k_smallest_diffs
    score = 0
    for path,_ in image_paths:
        if name in path:
            score += 1

    return score/ k

def showImages(images):
    h,w = images[0].height, images[0].width
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x,y))
        x+= w

    dst.show()


def show_example(files, dist_dict, k = 7, file = None):

    if file is None:
        file = random.choice(files)

    row = list(dist_dict[file].items())
    k_smallest_diffs = sorted(row, key=lambda x: x[1])[0:k ]
    image_paths = [(file,0)] + k_smallest_diffs
    out = []
    imgs = []
    for path,diff in image_paths:
        img = Image.open(path)
        out.append(path)
        imgs.append(img)

    showImages(imgs)

generate_dict = True
outfile = "1000_images"
scores, labels, files = load_data(outfile)
if generate_dict:
    diffs = dist_dict(scores, files, outfile)
else:
    diffs = np.load("../Dataset_CNN/data/" + outfile + "_diffs.npy", allow_pickle=True)

accuracies = []
max_ac = 0
k = 4
best = ""
random.shuffle(files)
for i in range(500):
    file =random.choice(files)
    ac = get_accuracy(file, diffs, k)
    accuracies.append(ac)

    if ac >= max_ac:
        best = file
        max_ac = ac

print(np.mean(accuracies))
print({ac : accuracies.count(ac) for ac in accuracies})
print(best)
print(max_ac)
show_example(files, diffs, k=k, file=best)

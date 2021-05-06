
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch.tensor as tensor
from tqdm import tqdm
from PIL import Image
import torch
from Dataset_CNN.EmbeddingNetwork import EmbeddingNetwork
from Dataset_CNN.CNN import data_transforms, ScoreFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()) )
else:
    print('Using CPU device.')

DATA_FOLDER = "../../uob_image_set_1000"
BATCH_SIZE = 50

def load_data(outfile):
    image_data = ScoreFolder(root=DATA_FOLDER, transform=data_transforms["val"])
    data_loader = torch.utils.data.DataLoader(image_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    checkpoint = torch.load("../Dataset_CNN/data/100_images.pth")

    model = EmbeddingNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)
    model.eval()

    all_embs = []
    all_names = []

    for step, (batch_imgs, batch_labels, batch_paths) in tqdm(enumerate(data_loader)):
        batch_names = [path.split("\\")[-1].replace(".jpg", "") for path in batch_paths]
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        embs = model(batch_imgs)
        embs = embs.detach().cpu().numpy()

        all_embs.extend(embs)
        all_names.extend(batch_names)


    return np.array(all_embs), all_names

"""
Random Example
    pick random pic
    let n be number of pics in folder
    Get n - nearest pics
    
"""

def dist_dict(embeddings, names):
    s = embeddings.shape
    dist_dict = {k : {} for k in names}

    for i,f1 in tqdm(enumerate(names)):
        for j,f2 in enumerate(names):
            if i != j:
                f1_embedding = embeddings[i]
                f2_embedding = embeddings[j]

                dist = np.sum(abs(f1_embedding - f2_embedding))

                dist_dict[f1][f2] = dist_dict[f2][f1] = dist

    return dist_dict

def get_accuracy(name, dist_dict, k = 5):
    folder_name = name.split("_")[0]
    row = list(dist_dict[name].items())
    k_smallest_diffs = sorted(row, key=lambda x: x[1])[0:k]
    image_names =  k_smallest_diffs
    score = 0
    for other_names,_ in image_names:
        if folder_name in other_names:
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


def show_example(names, dist_dict, k = 7, name = None):

    if names is None:
        name = random.choice(names)

    folder = lambda name : name.split("_")[0]
    row = list(dist_dict[name].items())
    k_smallest_diffs = sorted(row, key=lambda x: x[1])[0:k]
    image_names = [(name,0)] + k_smallest_diffs
    image_paths = [(DATA_FOLDER + "/" + folder(name) + "/" + name + ".jpg",dist) for name,dist in image_names]
    out = []
    imgs = []
    for path,diff in image_paths:
        img = Image.open(path)
        out.append(path)
        imgs.append(img)

    showImages(imgs)


def test_model():
    generate_dict = True
    outfile = "1000_images"
    embs, names = load_data(outfile)
    if generate_dict:
        diffs = dist_dict(embs, names)
        np.save("../Dataset_CNN/data/" + outfile + "_diffs.npy", diffs)
    else:
        diffs = np.load("../Dataset_CNN/data/" + outfile + "_diffs.npy", allow_pickle=True).item()

    accuracies = []
    max_ac = 0
    k = 5
    best = ""
    random.shuffle(names)
    for i in range(400):
        name =names[i]
        ac = get_accuracy(name, diffs, k)
        accuracies.append(ac)

        if ac >= max_ac:
            best = name
            max_ac = ac

    print(np.mean(accuracies))
    print({ac : accuracies.count(ac) for ac in accuracies})
    print(best)
    print(max_ac)
    show_example(names, diffs, k=k, name=best)

def loss_graphs():
    df = pd.read_csv("../Dataset_CNN/data/1000_images_losses.csv")
    df["error_diff"] = df['Neg_Loss'] - df['Pos_Loss']
    import seaborn as sns
    fig,axs = plt.subplots(1,3)
    fig.set_size_inches(30,10)
    sns.lineplot(x = df["Epochs"], y = df["Pos_Loss"], ax=axs[0]).set_title("Pos_Loss")
    sns.lineplot(x = df["Epochs"], y = df["Neg_Loss"], ax=axs[1]).set_title("Neg_loss")
    sns.lineplot(x = df["Epochs"], y = df["error_diff"], ax=axs[2]).set_title("Error Diff")

    plt.show()

loss_graphs()
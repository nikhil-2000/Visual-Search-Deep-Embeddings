from __future__ import absolute_import
import sys, os

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)


T_G_WIDTH = 100
T_G_HEIGHT = 134
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

usagemessage = 'Usage: \n\t -learn <Train Folder> <embedding size> <batch size> <num epochs> <output model file> \n\t -extract <Model File> <Input Image Folder> <Output File Prefix (TXT)> <tsne perplexity (optional)>\n\t\tBuilds and scores a triplet-loss embedding model.'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from Dataset_CNN.EmbeddingNetwork import EmbeddingNetwork
import PIL

# Misc. Necessities
import sys
import numpy as np
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# visualizations
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from Dataset_CNN.triples_dataset import ClothesFolder

# correct "too many files" error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

np.random.seed(T_G_SEED)
torch.manual_seed(T_G_SEED)
random.seed(T_G_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()) )
else:
    print('Using CPU device.')

# Image Transforms for pre-trained model.
# Normalization parameters taken from documentation for pre-trained model.
input_size = T_G_WIDTH
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

def set_parameter_requires_grad(model, feature_extracting):
    if (feature_extracting):
        for param in model.parameters():
            param.requires_grad = False






class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return(x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative_a = self.calc_euclidean(anchor, negative)
        distance_negative_b = self.calc_euclidean(positive, negative)

        losses = torch.relu(distance_positive - (distance_negative_a + distance_negative_b)/2.0 + self.margin)

        return losses.mean()

class ScoreFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super(ScoreFolder, self).__init__(root=root, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, label = super(ScoreFolder, self).__getitem__(index=index)

        # label may be meaningless if the data isn't labeled,
        # but it can simply be ignored.
        return img, label, self.samples[index][0]

def learn(argv):
    # <Train Folder> <embedding size> <batch size> <num epochs> <output model file root>
    if len(argv) < 5:
        print(usagemessage)
        return

    in_t_folder = argv[0]
    emb_size = int(argv[1])
    batch = int(argv[2])
    numepochs = int(argv[3])
    outpath = argv[4]

    margin = 1.0

    print('Triplet embeding training session. Inputs: ' + in_t_folder + ', ' + str(emb_size) + ', ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)

    train_ds = ClothesFolder(root=in_t_folder, transform=data_transforms['train'])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=1)

    # Allow all parameters to be fit
    model = EmbeddingNetwork(freeze_params=False)

    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)  # send model to GPU

    optimizer = optim.Adadelta(model.parameters())  # optim.Adam(model.parameters(), lr=0.01)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    criterion = TripletLoss(margin=margin)

    model.train()

    # let invalid epochs pass through without training
    if numepochs < 1:
        epoch = 0
        loss = 0

    for epoch in tqdm(range(numepochs), desc="Epochs"):
        running_loss = []
        train_ds.pick_batches(100)
        train_ds.calc_distances()
        for step, (anchor_img, positive_img, negative_img) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)  # send image to GPU
            positive_img = positive_img.to(device)  # send image to GPU
            negative_img = negative_img.to(device)  # send image to GPU

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            del anchor_img
            del positive_img
            del negative_img

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())

        train_ds.reset_remaining_folders()
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, np.mean(running_loss)))

    torch.save({
        'emb_size': emb_size,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimzier_state_dict': optimizer.state_dict(),
        'loss': loss
    }, outpath + '.pth')

    return


def extract(argv):
    if len(argv) < 3:
        print(usagemessage)
        return

    # have numpy print vectors to single lines
    np.set_printoptions(linewidth=np.inf)

    modelfile = argv[0]
    imgpath = argv[1]
    outfile = argv[2]

    # optionally support t-sne visualization
    tsne = 0
    if len(argv) >= 3:
        tsne = int(argv[3])

    checkpoint = torch.load(modelfile)

    model = EmbeddingNetwork(checkpoint['emb_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)
    model.eval()

    score_ds = ScoreFolder(imgpath, transform=data_transforms['val'])
    score_loader = DataLoader(score_ds, batch_size=1, shuffle=False, num_workers=1)

    results = []
    paths = []
    labels = []

    with torch.no_grad():
        for step, (img, label, path) in enumerate(tqdm(score_loader)):
            results.append(model(img.to(device)).cpu().numpy())
            paths.append(path)
            labels.append(label)

    with open(outfile + '_files.txt', 'w') as f:
        for item in paths:
            f.write("%s\n" % item)
        f.close()

    with open(outfile + '_labels.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
        f.close()

    with open(outfile + '_scores.txt', 'w') as f:
        for item in results:
            # f.write("%s\n" % str(item[0]))
            np.savetxt(f, item[0], newline=' ')
            f.write("\n")
        f.close()

    if (tsne > 0):
        scores_a = np.vstack(results)
        labels_a = np.vstack(labels)

        print('labels shape:' + str(labels_a.shape))
        sys.stdout.flush()

        tsne = TSNE(n_components=2, verbose=1, perplexity=tsne, n_iter=300, init='pca', learning_rate=10)
        tsne_results = tsne.fit_transform(scores_a)

        df_subset = {}
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['y'] = labels_a[:, 0]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("hls", len(Counter(labels_a[:, 0]).keys())),
            data=df_subset,
            hue="y",
            legend="brief",
            alpha=1.0
        )
        plt.savefig(outfile + '_tsne.png')

    return


def main(argv):

    if len(argv) < 2:
        print(usagemessage)
        return

    if 'learn' in argv[0]:
        learn(argv[1:])
    elif 'extract' in argv[0]:
        extract(argv[1:])

    return

def get_args():
    print(usagemessage)
    print("Enter args:")
    args = input()
    args = args.split(" ")
    return args


# Main Driver
if __name__ == "__main__":
    # args = get_args()
    main(sys.argv[1:])
    # in_t_folder = "../../uob_image_set_100"
    # batch = 1
    # train_ds = ClothesFolder(root=in_t_folder, transform=data_transforms["train"])
    # train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=1)
    #
    # anchor,pos, neg = next(iter(train_loader))
    # fig, axs = plt.subplots(1,3)
    # print(anchor.shape)
    # axs[0].imshow(anchor[0].permute(1, 2, 0))
    # axs[1].imshow(pos[0].permute(1, 2, 0))
    # axs[2].imshow(neg[0].permute(1, 2, 0))
    # plt.show()
    #
    # print("DONE")

#-learn ../../uob_image_set_100 1000 10 20 data/triplet
#-extract data/triplet.pth ../../uob_image_set_100 data/triplet 1
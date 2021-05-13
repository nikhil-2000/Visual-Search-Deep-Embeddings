from __future__ import absolute_import

import datetime
import os
import sys

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

T_G_WIDTH = 100
T_G_HEIGHT = 100
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

usagemessage = 'Usage: \n\t -learn <Train Folder> <batch size> <num epochs> <output model file root> <search size> <stop_label_training> <margin> \n\t -extract <Model File> <Input Image Folder> <Output File Prefix (TXT)> <tsne perplexity (optional)>\n\t\tBuilds and scores a triplet-loss embedding model.'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Misc. Necessities
import sys
import numpy as np
import random

# visualizations
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from Dataset_CNN.triples_dataset import ClothesFolder,ScoreFolder
from Dataset_CNN.writing_to_projector import write_to_projector
from Dataset_CNN.EmbeddingNetwork import EmbeddingNetwork

# correct "too many files" error
import torch.multiprocessing

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GridSearchCV


torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

np.random.seed(T_G_SEED)
torch.manual_seed(T_G_SEED)
random.seed(T_G_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()))
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
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative_anchor = self.calc_euclidean(anchor, negative)
        distance_negative_positive = self.calc_euclidean(positive, negative)

        losses = torch.relu(distance_positive - distance_negative_anchor + self.margin)

        return losses.mean()



def learn(argv):
    # <Train Folder> <batch size> <num epochs> <output model file root> <search size> <stop_label_training> <margin>
    if len(argv) < 6:
        print(usagemessage)
        return


    in_t_folder = argv[0]
    assert os.path.isdir(in_t_folder)

    batch = int(argv[1])
    assert batch > 0, "Batch size should be more than 0"

    numepochs = int(argv[2])
    assert numepochs > 0, "Need more than " + str(numepochs)  + " epochs"

    outpath = argv[3]

    search_size = int(argv[4])
    assert search_size > 0, "Need larger search size than " + str(search_size)

    stop_label_training = 0 #int(argv[5])
    assert 0 <= stop_label_training <= numepochs, "Need to stop training labels at some point"

    margin = float(argv[6])
    assert 0 < margin, "Pick a margin greater than 0"




    if not os.path.isdir("Outputs"): os.mkdir("Outputs")
    if not os.path.isdir("Outputs/Images"): os.mkdir("Outputs/Images")
    if not os.path.isdir("Outputs/Embeddings"): os.mkdir("Outputs/Embeddings")

    print('Triplet embeding training session. Inputs: ' + in_t_folder + ', ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath + ', ' + str(search_size) + ', ' + str(stop_label_training) + ', ' + str(margin))

    train_ds = ClothesFolder(root=in_t_folder, transform=data_transforms['train'], margin=margin)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=1)

    # Allow all parameters to be fit
    model = EmbeddingNetwork(freeze_params=False)

    # model = torch.jit.script(model).to(device) # send model to GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    criterion = TripletLoss(margin=margin)
    model.train()

    # let invalid epochs pass through without training
    if numepochs < 1:
        epoch = 0
        loss = 0

    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + "_Epochs" + str(numepochs) + "_Datasize" + str(len(train_ds))
    writer = SummaryWriter(log_dir="runs/" + run_name)

    s = 0
    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances
        train_ds.pick_batches(search_size)
        train_ds.calc_distances()

        # Calc errors before training for this epoch and add to df
        positive_loss, negative_loss, avg_accuracy = train_ds.calculate_error_averages()

        if epoch > 0:
            writer.add_scalar("Epoch_Checks/Pos_Neg_Difference", negative_loss - positive_loss, epoch)
            writer.add_scalar("Epoch_Checks/Avg_Accuracy", avg_accuracy, epoch)

        losses = []
        for step, (anchor_img, positive_img, negative_img) in enumerate(
                tqdm(train_loader, desc="Training", leave=True, position=0)):
            anchor_img = anchor_img.to(device)  # send image to GPU
            positive_img = positive_img.to(device)  # send image to GPU
            negative_img = negative_img.to(device)  # send image to GPU

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            #Clears space on GPU I think
            del anchor_img
            del positive_img
            del negative_img
            #Triplet Loss !!! + Backprop
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

            batch_norm = torch.linalg.norm(anchor_out, ord = 1)
            embedding_norm = torch.mean(batch_norm)

            writer.add_scalar("Loss/triplet_loss", loss,s )
            writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

            batch_positive_loss = torch.mean(criterion.calc_euclidean(anchor_out,positive_out))
            batch_negative_loss = torch.mean(criterion.calc_euclidean(anchor_out,negative_out))
            writer.add_scalar("Other/Positive_Loss", batch_positive_loss, s)
            writer.add_scalar("Other/Negative_Loss", batch_negative_loss, s)
            writer.add_scalar("Other/Pos_Neg_Difference", batch_negative_loss - batch_positive_loss, s)

            s += batch

        # Initially use_labels is True
        # This aims to split up the data by labels initially
        # After variable epochs, it begins to train the view invariance
        if stop_label_training == epoch:
            train_ds.training_labels = False

        writer.add_scalar("Epoch_Checks/triplet_loss", np.mean(losses), epoch+1)

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, negative_loss - positive_loss))

        #Writes errors to pd to review while training

        #Saves model so that distances can be updated using new model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimzier_state_dict': optimizer.state_dict(),
            'loss': loss
        }, outpath + '.pth')

        train_ds.modelfile = outpath + '.pth'

    #Calculate distances one final time to get last error
    train_ds.calc_distances()
    positive_loss, negative_loss, avg_accuracy = train_ds.calculate_error_averages()


    writer.add_scalar("Epoch_Checks/Pos_Neg_Difference", negative_loss - positive_loss, epoch+1)
    writer.add_scalar("Epoch_Checks/Avg_Accuracy", avg_accuracy, epoch+1)



    write_to_projector(in_t_folder, "Outputs/Images", "Outputs/Embeddings","runs", run_name + "_projector", model)
    writer.flush()
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

def visualise(argv):
    in_t_folder = argv[0]
    model_path = argv[1]
    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    checkpoint = torch.load(model_path)

    model = EmbeddingNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)
    model.eval()

    write_to_projector(in_t_folder, "Outputs/Images", "Outputs/Embeddings","runs", run_name + "_projector", model)


def main(argv):
    if len(argv) < 2:
        print(usagemessage)
        return

    if 'learn' in argv[0]:
        learn(argv[1:])
    elif 'extract' in argv[0]:
        extract(argv[1:])
    elif "visualise" in argv[0]:
        visualise(argv[1:])
    else:
        print("Didn't select learn or extract")

    return


# Main Driver
if __name__ == "__main__":
    main(sys.argv[1:])


# -learn ../../uob_image_set_100 1000 10 20 data/triplet
# -extract data/triplet.pth ../../uob_image_set_100 data/triplet 1

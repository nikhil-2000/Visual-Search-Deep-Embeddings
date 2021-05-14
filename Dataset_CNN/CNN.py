from __future__ import absolute_import

import datetime
import os
import sys

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)


usagemessage = '''
Usage: 
\t -learn <Train Folder> <batch size> <num epochs> <output model file root> <search size> <margin> <validation_folder?> - Trains Model
\t -extract <Model File> <Input Image Folder> <Output File Prefix (TXT)> <tsne perplexity (optional)> - Builds and scores a triplet-loss embedding model.
\t -visualise <Image Folder> <Path File> - Writes embeddings to projector using given model weights
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

from Dataset_CNN.triples_dataset import ClothesFolder, ScoreFolder
from Dataset_CNN.writing_to_projector import write_to_projector
from Dataset_CNN.EmbeddingNetwork import EmbeddingNetwork

# correct "too many files" error
import torch.multiprocessing

from torch.utils.tensorboard import SummaryWriter

from Dataset_CNN.constants import *

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

np.random.seed(T_G_SEED)
torch.manual_seed(T_G_SEED)
random.seed(T_G_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir("Outputs"): os.mkdir("Outputs")
if not os.path.isdir("Outputs/Images"): os.mkdir("Outputs/Images")
if not os.path.isdir("Outputs/Embeddings"): os.mkdir("Outputs/Embeddings")

def output_device():
    if device.type == "cuda":
        print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Using CPU device.')


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
    # <Train Folder> <batch size> <num epochs> <output model file root> <search size> <margin> <validation_set>
    if len(argv) < 6:
        print(usagemessage)
        return

    in_t_folder = argv[0]
    assert os.path.isdir(in_t_folder), "Pick valid image directory"

    batch = int(argv[1])
    assert batch > 0, "Batch size should be more than 0"

    numepochs = int(argv[2])
    assert numepochs > 0, "Need more than " + str(numepochs) + " epochs"

    outpath = argv[3]

    search_size = int(argv[4])
    assert search_size > 0, "Need larger search size than " + str(search_size)

    margin = float(argv[5])
    assert 0 < margin, "Pick a margin greater than 0"

    phases = ["train"]
    doValidation = len(argv) > 6
    if doValidation:
        validation_folder = argv[6]
        assert os.path.isdir(validation_folder), "Pick valid image directory"
        phases.append("validation")

    print('Triplet embeding training session. Inputs: ' + in_t_folder + ', ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath + ', ' + str(
        search_size) + ', ' + ', ' + str(margin))

    print("Validation will happen ? ", doValidation)

    train_ds = ClothesFolder(root=in_t_folder, transform=data_transforms['train'], margin=margin)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=torch.cuda.device_count() * 4)

    datasets = {"train": train_ds}
    data_loaders = {"train": train_loader}
    folder_paths = {"train" : in_t_folder}
    if doValidation:
        validation_ds = ClothesFolder(root=validation_folder, transform=data_transforms['val'], margin=margin)
        validation_loader = DataLoader(validation_ds, batch_size=batch, shuffle=True,
                                       num_workers=torch.cuda.device_count() * 4)
        datasets["validation"] = validation_ds
        data_loaders["validation"] = validation_loader
        folder_paths["validation"] = validation_folder

    # Allow all parameters to be fit
    model = EmbeddingNetwork(freeze_params=False)

    # model = torch.jit.script(model).to(device) # send model to GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    criterion = TripletLoss(margin=margin)

    # let invalid epochs pass through without training
    if numepochs < 1:
        epoch = 0
        loss = 0

    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + "_Epochs" + str(numepochs) + "_Datasize" + str(
        len(train_ds))
    writer = SummaryWriter(log_dir="runs/" + run_name)

    steps = {"train": 0, "validation": 0}
    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances

        for phase in phases:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            dataset, data_loader = datasets[phase], data_loaders[phase]
            dataset.pick_batches(search_size)
            dataset.calc_distances()

            # Calc errors before training for this epoch and add to df
            positive_loss, negative_loss, avg_accuracy = dataset.calculate_error_averages()

            if epoch > 0:
                writer.add_scalar("Epoch_Pos_Neg_Difference/" + phase, negative_loss - positive_loss, epoch)
                writer.add_scalar("Epoch_Avg_Accuracy/" + phase, avg_accuracy, epoch)

            losses = []
            for step, (anchor_img, positive_img, negative_img) in enumerate(
                    tqdm(data_loader, desc=phase, leave=True, position=0)):

                anchor_img = anchor_img.to(device)  # send image to GPU
                positive_img = positive_img.to(device)  # send image to GPU
                negative_img = negative_img.to(device)  # send image to GPU

                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)
                # Clears space on GPU I think
                del anchor_img
                del positive_img
                del negative_img
                # Triplet Loss !!! + Backprop
                loss = criterion(anchor_out, positive_out, negative_out)

                optimizer.zero_grad()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                losses.append(loss.cpu().detach().numpy())

                # batch_norm = torch.linalg.norm(anchor_out, ord = 1, dim= 1)
                # embedding_norm = torch.mean(batch_norm)
                # writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

                writer.add_scalar("triplet_loss/" + phase, loss, steps[phase])

                batch_positive_loss = torch.mean(criterion.calc_euclidean(anchor_out, positive_out))
                batch_negative_loss = torch.mean(criterion.calc_euclidean(anchor_out, negative_out))
                writer.add_scalar("Other/Positive_Loss/" + phase, batch_positive_loss, steps[phase])
                writer.add_scalar("Other/Negative_Loss/" + phase, batch_negative_loss, steps[phase])
                writer.add_scalar("Pos_Neg_Difference/" + phase, batch_negative_loss - batch_positive_loss,
                                  steps[phase])

                steps[phase] += batch

            writer.add_scalar("Epoch_triplet_loss/" + phase, np.mean(losses), epoch + 1)

            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, negative_loss - positive_loss))

            # Saves model so that distances can be updated using new model
            if phase == "train":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimzier_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, outpath + '.pth')

            dataset.modelfile = outpath + '.pth'

    # Calculate distances one final time to get last error
    epoch += 1
    for phase, dataset in datasets.items():
        dataset.calc_distances()
        positive_loss, negative_loss, avg_accuracy = dataset.calculate_error_averages()

        writer.add_scalar("Epoch_Pos_Neg_Difference/" + phase, negative_loss - positive_loss, epoch)
        writer.add_scalar("Epoch_Avg_Accuracy/" + phase, avg_accuracy, epoch)

        write_to_projector(folder_paths[phase], "Outputs/Images", "Outputs/Embeddings", "runs",
                           run_name + "_" + phase + "_projector", model)
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

    if len(argv) < 2:
        print(usagemessage)

    in_t_folder = argv[0]
    model_path = argv[1]
    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    checkpoint = torch.load(model_path)

    model = EmbeddingNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)
    model.eval()

    write_to_projector(in_t_folder, "Outputs/Images", "Outputs/Embeddings", "runs", run_name + "_projector", model)


def main(argv):
    output_device()
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

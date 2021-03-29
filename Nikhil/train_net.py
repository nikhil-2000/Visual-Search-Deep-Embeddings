import numpy as np

from Nikhil.CNN import Net
from torchvision import transforms
from Nikhil.triples_dataset import ClothesDataset

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as f

import tqdm

images_path = 'D:\My Docs/University\Applied Data Science\Project/uob_image_set'

BATCH_SIZE = 20
emb_size = 10

net = Net(emb_size=10)

transform = transforms.Resize((67, 50))
dataset = ClothesDataset(images_path, 1500, transform=transform, sample_for_negatives= 1)
train_data = DataLoader(dataset, batch_size=20, shuffle=True)

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.TripletMarginLoss()

# VAL_PCT = 0.1
# val_size = int(len(X) * VAL_PCT)
# print(val_size)

print(len(train_data))

# %%

EPOCHS = 3

net.train()
epoch = 0
for epoch in range(EPOCHS):
    running_loss = []
    for step, (anchor_img, positive_img, negative_img) in enumerate(train_data):
        optimizer.zero_grad()
        print(anchor_img.shape)
        anchor_out = net(anchor_img)
        positive_out = net(positive_img)
        negative_out = net(negative_img)

        loss = loss_function(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.cpu().detach().numpy())
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, EPOCHS, np.mean(running_loss)))

outpath = "D:\My Docs/University\Applied Data Science\Project\Visual-Search-Deep-Embeddings/Nikhil"

torch.save({
    'emb_size': emb_size,
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimzier_state_dict': optimizer.state_dict(),
    'loss': loss
}, outpath + '.pth')

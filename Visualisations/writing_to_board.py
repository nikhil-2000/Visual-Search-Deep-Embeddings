from __future__ import absolute_import

import os
import sys

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

from tqdm import tqdm
from torchvision import transforms
import torch.cuda
from Dataset_CNN.CNN import EmbeddingNetwork, ScoreFolder, data_transforms
from Visualisations.DF import DeepFeatures

BATCH_SIZE = 50
DATA_FOLDER = r'../../uob_image_set_1000'
IMGS_FOLDER = './Outputs/Images'
EMBS_FOLDER = './Outputs/Embeddings'
TB_FOLDER = './Outputs/Tensorboard'
EXPERIMENT_NAME = 'UOB_IMAGE_SET_VIS'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()) )
else:
    print('Using CPU device.')

def stack(tensor, times=3):
  return(torch.cat([tensor]*times, dim=0))

image_data = ScoreFolder(root = DATA_FOLDER, transform=data_transforms["val"])
data_loader = torch.utils.data.DataLoader(image_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


checkpoint = torch.load("D:\My Docs/University\Applied Data Science\Project\Visual-Search-Deep-Embeddings/1000_images.pth")

model = EmbeddingNetwork(checkpoint['emb_size'])
model.load_state_dict(checkpoint['model_state_dict'])
# model = torch.jit.script(model).to(device) # send model to GPU
model = model.to(device)
model.eval()



DF = DeepFeatures(model = model,
                  imgs_folder = IMGS_FOLDER,
                  embs_folder = EMBS_FOLDER,
                  tensorboard_folder = TB_FOLDER,
                  experiment_name=EXPERIMENT_NAME)

all_imgs = None
all_embs = None
all_names = None

for step, (batch_imgs, batch_labels, batch_paths) in tqdm(enumerate(data_loader)):
    batch_names = [path.split("\\")[-1].replace(".jpg", "") for path in batch_paths]
    batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
    embs = model(batch_imgs)


    # print("Input images: " + str(batch_imgs.shape))
    # print("Embeddings: " + str(embs.shape))
    # first_img = batch_imgs[0].to(device)
    # plt.imshow(first_img.permute(1, 2, 0).cpu())
    # plt.show()

    DF.write_embeddings(x = batch_imgs.to(device), labels = batch_names, outsize=(100, 100))

DF.create_tensorboard_log()

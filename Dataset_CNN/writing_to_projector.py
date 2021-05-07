from __future__ import absolute_import

import os
import sys

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

from tqdm import tqdm
import torch.cuda
from Dataset_CNN.triples_dataset import ScoreFolder, data_transforms
from Dataset_CNN.ProjectorObject import DeepFeatures


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_to_projector(DATA_FOLDER,IMGS_FOLDER, EMBS_FOLDER,TB_FOLDER,EXPERIMENT_NAME, model):
    image_data = ScoreFolder(root=DATA_FOLDER, transform=data_transforms["val"])
    data_loader = torch.utils.data.DataLoader(image_data,
                                              batch_size=50,
                                              shuffle=True)


    # model = torch.jit.script(model).to(device) # send model to GPU
    model = model.to(device)
    model.eval()

    projector = DeepFeatures(model=model,
                      imgs_folder=IMGS_FOLDER,
                      embs_folder=EMBS_FOLDER,
                      tensorboard_folder=TB_FOLDER,
                      experiment_name=EXPERIMENT_NAME)


    for step, (batch_imgs, batch_labels, batch_paths) in tqdm(enumerate(data_loader)):
        batch_names = [path.split("\\")[-1].replace(".jpg", "") for path in batch_paths]
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        embs = model(batch_imgs)


        # print("Input images: " + str(batch_imgs.shape))
        # print("Embeddings: " + str(embs.shape))
        # first_img = batch_imgs[0].to(device)
        # plt.imshow(first_img.permute(1, 2, 0).cpu())
        # plt.show()

        projector.write_embeddings(x = batch_imgs.to(device), labels = batch_names, outsize=(100, 100))

    projector.create_tensorboard_log()

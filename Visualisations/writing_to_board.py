from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.cuda
from Dataset_CNN.CNN import EmbeddingNetwork, ScoreFolder
import numpy as np
from Visualisations.DF import DeepFeatures
import matplotlib.pyplot as plt

BATCH_SIZE = 100
DATA_FOLDER = r'../../uob_image_set_10'
IMGS_FOLDER = './Outputs/Images'
EMBS_FOLDER = './Outputs/Embeddings'
TB_FOLDER = './Outputs/Tensorboard'
EXPERIMENT_NAME = 'UOB_IMAGE_SET_VIS'

T_G_WIDTH = 50
T_G_HEIGHT = 50
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

input_size = T_G_WIDTH

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((T_G_HEIGHT, T_G_WIDTH)),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

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


checkpoint = torch.load("..\Dataset_CNN/data/triplet.pth")

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

    if all_names is None:
        all_names = batch_names
    else:
        all_names.extend(batch_names)


    DF.write_embeddings(x = batch_imgs.to(device), labels = batch_names, outsize=(T_G_HEIGHT, T_G_HEIGHT))


DF.create_tensorboard_log(all_names)

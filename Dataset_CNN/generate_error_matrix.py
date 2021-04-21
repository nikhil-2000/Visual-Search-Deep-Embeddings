import os

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from tqdm import tqdm, trange
import pickle


def get_images(path, lim  = None):
    all_images = []
    num_img_list = []
    folders = os.listdir(path)
    if lim is None:
        lim = len(folders)

    folder_count = 0
    print("Loading Images")
    for folder in tqdm(folders):
        path_folder = os.path.join(path, folder)
        inside = os.listdir(path_folder)
        inside = [os.path.join(path_folder,i) for i in inside]
        num_img_list.append(len(inside))
        all_images.extend(inside)
        folder_count += 1

        if folder_count > lim:
            return all_images, num_img_list

    print()
    return all_images, num_img_list

# the path where the folder is

# %%

# using the mean and std values form the ImageNet data for which it was pretrained
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

def preprocess_and_batch_image(image_path):
    img = Image.open(image_path)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def feed_to_network(image_list, network):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Using CPU device.')

    output_array = []
    print("Feeding Batches")
    network.to(device)
    for i in trange(len(image_list)):
        input_batch = preprocess_and_batch_image(image_list[i])
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = network(input_batch)
            cpu_tensor = output.detach().cpu()
            pos = cpu_tensor[0].tolist()
            output_array.append(pos)

    output_array = np.asarray(output_array)
    print()
    return output_array



def get_path_list(path, lim):
    all_paths = []
    folders = os.listdir(path)
    for folder in folders:
        path_folder = os.path.join(path, folder)
        inside = os.listdir(path_folder)
        inside = [i.split(".")[0] for i in inside]
        all_paths.extend(inside)

        if len(all_paths) >= lim:
            return all_paths

    return all_paths



def get_diff_dicts(embeddings, image_names):
    n = len(image_names)
    class_to_idx = {name : i for i,name in enumerate(image_names)}
    idx_to_class = {i : name for i,name in enumerate(image_names)}
    emb_diff = dict([(name, {}) for name in image_names])
    pos_diff = dict([(name, {}) for name in image_names])

    print()
    print("Calculating Diffs...")
    for i in trange(0, n):
        folder_name = image_names[i].split("_")[0]
        for j in range(0, i):
            comparison_name = image_names[j].split("_")[0]
            emb_1 = embeddings[i]
            emb_1_key = image_names[i]
            emb_2 = embeddings[j]
            emb_2_key = image_names[j]
            diff = np.linalg.norm(emb_1 - emb_2)

            if folder_name != comparison_name:
                emb_diff[emb_1_key][emb_2_key] = diff
                emb_diff[emb_2_key][emb_1_key] = diff
            else:
                pos_diff[emb_1_key][emb_2_key] = diff
                pos_diff[emb_2_key][emb_1_key] = diff


    return [emb_diff, pos_diff, class_to_idx, idx_to_class]

def setup_model():
    resnet = models.resnet152(pretrained=True)
    # resnet = models.resnet152(pretrained=True)
    # freeze all base layers; Parameters of newly constructed modules have requires_grad=True by default
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = Identity()  # Remove the prediction head
    return resnet


class Identity(torch.nn.Module):
    def init(self):
        super(Identity, self).init()

    def forward(self, x):
        return x




# images, num_images = load_images_from_folder(path, 100, as_tensor=False)
def generate_matrix(path,name):
    images, num_images = get_images(path)
    resnet = setup_model()
    # list_input_batch, list_input_tensor = preprocess_and_batch(images)
    network_output = feed_to_network(images, resnet)

    all_paths = get_path_list(path, sum(num_images))
    negatives, positives, class_to_idx, idx_to_class = get_diff_dicts(network_output,all_paths )
    del resnet
    del network_output

    dir = "data/" + name
    os.mkdir(dir)
    os.mkdir(dir + '/negatives')
    os.mkdir(dir + '/positives')
        # Pickle the 'data' dictionary using the highest protocol available.
    for name,diffs in negatives.items():
        with open(dir + "/negatives/" + name + ".pickle", "wb") as f:
            pickle.dump(diffs, f, pickle.HIGHEST_PROTOCOL)

    for name,diffs in positives.items():
        with open(dir + "/positives/" + name + ".pickle", "wb") as f:
            pickle.dump(diffs, f, pickle.HIGHEST_PROTOCOL)

    # with open(dir + '/name_to_idx.pickle', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(class_to_idx, f, pickle.HIGHEST_PROTOCOL)
    #
    # with open(dir + '/idx_to_name.pickle', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(idx_to_class, f, pickle.HIGHEST_PROTOCOL)




# def feed_batch_to_network(list_input_batch, network):
#     output_array = []
#     print("Feeding Batches")
#     network.to('cuda')
#     for i in trange(len(list_input_batch)):
#         if torch.cuda.is_available():
#             input_batch = list_input_batch[i].to('cuda')
#         else:
#             input_batch = list_input_batch[i].to('cpu')
#
#         with torch.no_grad():
#             output = network(input_batch)
#             cpu_tensor = output.cpu()
#             pos = cpu_tensor[0].tolist()
#             output_array.append(pos)
#
#     output_array = np.asarray(output_array)
#     print()
#     return output_array

# def load_images_from_folder(folder, end, as_tensor=False):
#     images = []
#     num_img = []
#     file_count = 0
#     dirnames = []
#
#     for _, dirnames, filenames in os.walk(folder):
#
#         if dirnames != []:
#             subfolders = dirnames
#         current_path = os.path.join(folder, subfolders[file_count])
#         num_img.append(len(os.listdir(current_path)))
#         for filename in os.listdir(current_path):
#             if as_tensor:
#                 img = cv2.imread(os.path.join(current_path, filename))
#                 if img is not None:
#                     img = torch.from_numpy(img).type(torch.uint8)
#                     images.append(img)
#
#             else:
#                 img = Image.open(os.path.join(current_path, filename))
#                 if img is not None:
#                     images.append(img)
#
#         file_count += 1
#
#         if file_count % 10 == 0:
#             print('number of folder done =', file_count, 'total number of images so far =', len(images))
#
#         if file_count == end:
#             return images, num_img
#
#     return images, num_img
#
# def preprocess_and_batch(image_list):
#     list_input_tensor = []
#     list_input_batch = []
#     print("Preprocessing ...")
#     for item in trange(len(image_list)):
#         img = Image.open(image_list[item])
#         input_tensor = preprocess(img)
#         input_batch = input_tensor.unsqueeze(0)
#         list_input_tensor.append(input_tensor)
#         list_input_batch.append(input_batch)
#
#     print()
#     return list_input_batch, list_input_tensor
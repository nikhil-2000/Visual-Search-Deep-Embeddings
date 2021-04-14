import random

import numpy as np
from PIL import Image, ImageStat, ImageChops
import os
from torchvision import transforms
import cv2
from tqdm import tqdm

h,w = 134,100
# h,w = 1333, 1000
transform = transforms.Resize((h,w))
images_path = "../../uob_image_set_100"

def get_fft(img):

    gray = np.array(img.convert("LA"))
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

def full_path(dir_name):
    image_folder = os.path.join(images_path, dir_name)

    img_name = os.listdir(image_folder)[0]
    return os.path.join(image_folder, img_name)

def get_images():
    chosen = os.listdir(images_path)
    print("Loading Images...")
    imgs = [Image.open(full_path(k)) for k in tqdm(chosen)]

    print()
    print("Transforming Images")
    imgs = [transform(i) for i in tqdm(imgs)]
    print()
    print("LOADED")
    return imgs,chosen

def get_fourier_matrix(images, image_names):
    print("Calculating fft for images...")
    fft_images = [get_fft(i) for i in tqdm(images)]
    n = len(images)
    fft_diff = dict([(name, {}) for name in image_names])
    matrix = np.zeros((n,n))

    print()
    print("Calculating Diffs...")
    for i in tqdm(range(0, n)):
        for j in range(0, i):

            if i != j:
                fft_1 = fft_images[i]
                fft_1_name = image_names[i]
                fft_2 = fft_images[j]
                fft_2_name = image_names[j]


                diff = np.sum(abs(fft_1 - fft_2))
                fft_diff[fft_1_name][fft_2_name] = diff
                fft_diff[fft_2_name][fft_1_name] = diff
                matrix[i][j] = diff
                matrix[j][i] = diff

    fft_min, fft_max, mean = get_measures(matrix)

    transformed = (matrix - fft_min) / (fft_max - fft_min)
    return fft_diff, transformed

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def get_col(image):
    cropped = trim(image)
    if cropped is None or min(cropped.size) < 20:
        cropped = image

    median = ImageStat.Stat(cropped).median
    if sum(median)/3 > 200:
        cols = [tup for tup in cropped.getcolors(maxcolors=h*w) if tup[1] != (255,255,255)]

        cols = sorted(cols, key=lambda x: x[0], reverse=True)
        if cols:
            reds = [col[0] for _, col in cols]
            blues = [col[1] for _, col in cols]
            greens = [col[2] for _, col in cols]

            median = [np.median(cs) for cs in [reds,blues,greens]]
            median = [int(i) for i in median]

    return median

def get_colour_matrix(images, image_names):
    print("Getting Colours...")
    col_images = [get_col(i) for i in tqdm(images)]
    n = len(images)
    col_diff = dict([(name, {}) for name in image_names])
    matrix = np.zeros((n, n))

    print()
    print("Calculating Diffs...")
    for i in tqdm(range(0, n)):
        for j in range(0, i):

            if i != j:
                col_1 = np.array(col_images[i])
                col_2 = np.array(col_images[j])
                col_1_name, col_2_name = image_names[i] , image_names[j]


                diff = np.sum(abs(col_1 - col_2))
                col_diff[col_1_name][col_2_name] = col_diff[col_2_name][col_1_name] = diff
                matrix[i][j] = matrix[j][i] = diff

    col_min, col_max, mean = get_measures(matrix)

    transformed = (matrix - col_min) / (col_max - col_min)

    return col_diff, transformed

def get_measures(w_matrix):
    w_matrix = w_matrix.flatten()
    above_zero_idxs = np.where(w_matrix > 0)
    above_zero = w_matrix[above_zero_idxs]
    min_w = np.amin(above_zero)
    max_w = np.amax(above_zero)
    average = np.mean(above_zero)
    measures = [min_w, max_w, average]
    return measures


def convert_to_dict(matrix, names):
    n = len(names)
    diff_dict = dict([(name, {}) for name in names])
    for i in tqdm(range(0, n)):
        for j in range(0, i):

            if i != j:

                name_1, name_2 = names[i], names[j]
                diff_dict[name_1][name_2] = matrix[i][j]
                diff_dict[name_2][name_1] = matrix[i][j]

    return diff_dict

def generate_matrices(images,image_names):
    fft_diff_dict, fft_matrix = get_fourier_matrix(images, image_names)
    col_diff_dict, col_matrix = get_colour_matrix(images, image_names)
    #Saves Transformed Matrices
    np.save("fft_diff.npy", fft_matrix)
    np.save("col_diff.npy", col_matrix)

def generate_error_matrix():
    fft_matrix = np.load("fft_diff.npy", allow_pickle = True)
    col_matrix = np.load("col_diff.npy", allow_pickle = True)

    scaler = 1
    error_matrix = scaler * fft_matrix + col_matrix
    error_dict = convert_to_dict(error_matrix, image_names)

    np.save("error_diff.npy", error_dict)

def get_closest(image_names, error_diff, idx, k):
    class_name = image_names[idx]
    row = list(error_diff[class_name].items())
    k_smallest_idx = sorted(row, key=lambda x: x[1])[1:k + 1]
    return k_smallest_idx

def add_colour_below(images):
    joint_images = []

    for im in images:
        w,h = im.size
        mode_col = tuple(get_col(im))
        col_img = Image.new('RGB', (w, h), mode_col)
        fft_img = Image.fromarray(get_fft(im))
        dst = Image.new('RGB', (w, 3 * h))
        dst.paste(im, (0, 0))
        dst.paste(col_img, (0, im.height))
        dst.paste(fft_img, (0, 2*im.height))
        joint_images.append(dst)

    return joint_images

def convert_names_to_images(images,image_names, closest):
    closest_images = []
    names = [i[0] for i in closest]
    for name in names:
        img_idx = image_names.index(name)
        closest_images.append(images[img_idx])

    return closest_images

def showImages(images):
    h = images[0].height
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x,y))
        x+= w

    dst.show()

def show_example(images,image_names,n = 1):
    error_diff = np.load("../error_net_diff.npy", allow_pickle = True).item()

    for i in range(n):
        idx = random.randint(0, 100)
        # idx = image_names.index("15049546")
        chosen_img = images[idx]
        print(image_names[idx])
        closest = get_closest(image_names, error_diff, idx, 10)
        closest_images = convert_names_to_images(images,image_names, closest)
        images_with_median_col = add_colour_below([chosen_img] + closest_images)
        showImages(images_with_median_col)


if __name__ == '__main__':

    images, image_names = get_images()
    generate_matrices(images,image_names)
    generate_error_matrix()
    # show_example(images,image_names,n = 10)


    # check = ["13888239","14342457","14713205","15146351"]
    # for i_name in check:
    #     path = "D:\My Docs/University\Applied Data Science\Project/uob_image_set/" + i_name + "/" + i_name + "_0.jpg"
    #     img = transform(Image.open(path))
    #     img_with_col = add_colour_below([img])
    #     img_with_col[0].show()

    #14455761
    #15923931
    #13888239
    #14342457
    #14713205
    #15146351

    # step = 20
    # for i in range(0, len(images), step):
    #     start = i
    #     batch = images[start: start + step]
    #     batch_col = add_colour_below(batch)
    #     showImages(batch_col)
    #     print(image_names[start: start + step])
    #     input()

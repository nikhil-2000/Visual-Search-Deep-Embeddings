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
images_path = '../../uob_image_set'

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
    print("LOADED")
    return imgs,chosen

def get_fourier_matrix(images, image_names):
    print("Calculating fft for images...")
    fft_images = [get_fft(i) for i in tqdm(images)]
    n = len(images)
    fft_diff = dict([(name, {}) for name in image_names])

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

    return fft_diff



def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def get_col(image):
    centre =  transforms.CenterCrop((30, 30))
    cropped = trim(image)
    if cropped is None:
        cropped = centre(image)

    return ImageStat.Stat(cropped).median

def get_colour_matrix(images, image_names):
    print("Getting Colours...")
    col_images = [get_col(i) for i in tqdm(images)]
    n = len(images)
    col_diff = dict([(name, {}) for name in image_names])

    print()
    print("Calculating Diffs...")
    for i in tqdm(range(0, n)):
        for j in range(0, i):

            if i != j:
                col_1 = np.array(col_images[i])
                col_1_name = image_names[i]
                col_2 = np.array(col_images[j])
                col_2_name = image_names[j]


                diff = np.sum(abs(col_1 - col_2))
                col_diff[col_1_name][col_2_name] = diff
                col_diff[col_2_name][col_1_name] = diff

    return col_diff

def get_measures(w_matrix):
    w_matrix = w_matrix.flatten()
    above_zero_idxs = np.where(w_matrix > 0)
    above_zero = w_matrix[above_zero_idxs]
    min_w = np.amin(above_zero)
    max_w = np.amax(above_zero)
    average = np.mean(above_zero)
    # variance = np.var(above_zero)
    measures = [min_w, max_w, average]
    return measures

def get_error_matrix(fft_diff_matrix, col_diff_matrix):
    fft_min, fft_max, fft_mean = get_measures(fft_diff_matrix)
    col_min, col_max, col_mean = get_measures(col_diff_matrix)
    transformed_fft = (fft_diff_matrix - fft_min) / (fft_max - fft_min)
    transformed_col = (col_diff_matrix - col_min) / (col_max - col_min)
    col_measures = get_measures(transformed_col)
    fft_measures = get_measures(transformed_fft)
    scaler =  2 * col_measures[2] / fft_measures[2]
    error_matrix =  scaler * transformed_fft + transformed_col
    return error_matrix

def generate_matrices():
    images, image_names = get_images()
    fft_diff_matrix = get_fourier_matrix(images, image_names)
    col_diff_matrix = get_colour_matrix(images, image_names)

    np.save("fft_diff.npy", fft_diff_matrix)
    np.save("col_diff.npy", col_diff_matrix)


def get_closest(images, fft_diff, idx, k):
    images = np.array(images)
    row = fft_diff[idx]
    k_smallest_idx = np.argsort(row)[1:k+1]
    return images[k_smallest_idx]

def add_colour_below(images):
    joint_images = []

    for im in images:
        w,h = im.size
        mode_col = tuple(get_col(im))
        col_img = Image.new('RGB', (w, h), mode_col)
        dst = Image.new('RGB', (w, 2 * h))
        dst.paste(im, (0, 0))
        dst.paste(col_img, (0, im.height))
        joint_images.append(dst)

    return joint_images

def showImages(images, h = h):

    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x,y))
        x+= w

    dst.show()


def show_example():
    images = get_images()
    fft_diff = np.load("fft_diff.npy")
    col_diff = np.load("col_diff.npy")
    error_matrix = get_error_matrix(fft_diff, col_diff)
    idx = random.randint(0, 1500)
    chosen_img = images[idx]
    closest = get_closest(images, error_matrix, idx, 10)
    images_with_median_col = add_colour_below([chosen_img] + list(closest))
    showImages(images_with_median_col, h = chosen_img.height * 2)

generate_matrices()



import numpy as np

n = 100
first_n = lambda x: x[:n]


embeddings = first_n(np.loadtxt("txt_scores.txt"))
with open('txt_files.txt') as f:
    files = f.read().split("\n")

with open('txt_labels.txt') as f:
    labels = f.read().split("\n")

files = first_n(files)
labels = first_n(labels)


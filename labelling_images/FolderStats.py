from __future__ import absolute_import
import sys, os

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

from torchvision.datasets import ImageFolder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FolderStats(ImageFolder):

    def __init__(self, root):
        super(FolderStats, self).__init__(root=root, transform=None)

        self.labels_to_folder, self.folder_to_labels = self.convert_to_dict(
            pd.read_csv("../labelling_images/labelled.csv"))



        self.images = {}
        for dir in self.classes:
            class_index = self.class_to_idx[dir]
            self.images[class_index] = []

        for s in self.samples:
            if s[1] in self.images.keys():
                self.images[s[1]].append(s[0])

        self.classdictsize = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdictsize[ci] = len(self.images[ci])

    def get_labels_dict(self):
        return {label : len(folders) for label,folders in self.labels_to_folder.items()}

    def convert_to_dict(self, labelled_df):
        #Creates the lookups for label -> folder and folder -> label
        label_to_folder = {}
        folder_to_label = {}
        for index, row in labelled_df.iterrows():
            label, folder = row["label"], str(row["folder"])
            if not (label in label_to_folder.keys()):
                label_to_folder[label] = []

            if folder in self.classes:
                label_to_folder[label].append(folder)
                folder_to_label[folder] = label



        return label_to_folder, folder_to_label

def multi_bar_plot(train_labels,val_labels):
    df_1 = set_up_df(train_labels, "train")
    df_2 = set_up_df(val_labels, "validation")

    all_data = pd.concat([df_1,df_2], ignore_index=True)
    g = sns.catplot(x = "Label", y = "Count", hue = "Set", kind = "bar", data=all_data)
    g.set_xticklabels(rotation=80)
    plt.show()

def set_up_df(label_count,set):
    cols = ["Label", "Count"]
    df = pd.DataFrame(label_count.items(), columns=cols)
    df["Set"] = set
    return df

def get_relative_counts(label_count):
    total = sum(label_count.values())
    return {k : v * 100 /total for k,v in label_count.items()}

def relative_labels_plot(train_labels, val_labels):

    relative_train = get_relative_counts(train_labels)
    relative_val = get_relative_counts(val_labels)

    df_1 = set_up_df(relative_train, "train")
    print(df_1.sum())
    df_2 = set_up_df(relative_val, "validation")
    print(df_2.sum())

    all_data = pd.concat([df_1,df_2], ignore_index=True)
    g = sns.lineplot(x='Label', y='Count', data=all_data,
                 hue="Set")

    g.set_xticklabels(g.get_xticklabels(), rotation=80)
    g.set(xlabel='Label', ylabel='Percentage')

    plt.show()

def main():
    train_set = FolderStats(r"../../uob_image_set_500")
    validation_set = FolderStats(r"../../uob_image_set_500_validation")
    train_labels = train_set.get_labels_dict()
    validation_labels = validation_set.get_labels_dict()

    # multi_bar_plot(train_labels, validation_labels)
    relative_labels_plot(train_labels, validation_labels)

if __name__ == '__main__':
    main()
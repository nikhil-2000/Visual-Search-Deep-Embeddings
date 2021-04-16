
with open("labelled.csv", 'r') as f:
    labelled_folders = f.read().split("\n")
    while '' in labelled_folders: labelled_folders.remove('')

    if labelled_folders is not None:
        labelled_folders = [line.split(",") for line in labelled_folders]
        labelled_folders.extend(labelled_folders)

with open("unlabelled.txt", 'r') as f:
    unlabelled_folders = f.read().split("\n")

while '' in unlabelled_folders: unlabelled_folders.remove('')

labelled_names = [t[0] for t in labelled_folders]
inBoth = set(labelled_names).union(set(unlabelled_folders))
print(len(set(labelled_names)))
print(len(unlabelled_folders))

with open("labelled.csv", 'r') as f:
    labelled_folders = f.read().split("\n")
    while '' in labelled_folders: labelled_folders.remove('')

    if labelled_folders is not None:
        labelled_folders = [line.split(",") for line in labelled_folders]
        # labelled_folders.extend(labelled_folders)

with open("unlabelled.txt", 'r') as f:
    unlabelled_folders = f.read().split("\n")

while '' in unlabelled_folders: unlabelled_folders.remove('')

labelled_names = [t[0] for t in labelled_folders]
count_2 = set([l for l in labelled_names if labelled_names.count(l) > 1])
inBoth = set(labelled_names).union(set(unlabelled_folders))
print(len(labelled_names))
print(len(unlabelled_folders))
print(len(count_2))
print(len(labelled_folders))


# for item in count_2:
#     all_labels = [line for line in labelled_folders if item == line[0]]
#     first_line = all_labels[0]
#     labelled_folders.remove(first_line)
#
# with open("labelled_dup.csv", "w") as f:
#     lines = [",".join(line) for line in labelled_folders]
#     text = "\n".join(lines)
#     f.write(text)

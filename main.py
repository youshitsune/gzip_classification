import gzip
import csv
import numpy as np

test_set = []
train_set = []
k = 4

class_index = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}

with open("train.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_set.append((row['Description'], row["Class Index"]))

with open("test.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_set.append((row["Description"], row["Class Index"]))

test_set = np.array(test_set[:10])
train_set = np.array(train_set[:100000])

for (x1, _) in test_set:
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []
    for (x2, _) in train_set:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)
    sorted_idx = np.argsort(np.array(distance_from_x1))
    top_k_class = train_set[sorted_idx[:k], 1]
    top_k_class = list(top_k_class)
    predict_class = max(set(top_k_class), key=top_k_class.count)
    print(f"{class_index[predict_class]} - {x1}")


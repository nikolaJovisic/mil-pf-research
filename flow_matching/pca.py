import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append("../head_training")

pkl_path = "/lustre/nj/cvpr2026/pickles/vindr-msl.pkl"

with open(pkl_path, "rb") as f:
    train_ds, valid_ds, test_ds = pickle.load(f)

train_x, train_y, train_w, train_group, train_instance = train_ds[0]
valid_x, valid_y, valid_w, valid_group, valid_instance = valid_ds[0]
test_x, test_y, test_w, test_group, test_instance = test_ds[0]

train_x_np = train_x.detach().cpu().numpy()

pca = PCA(n_components=8, svd_solver="randomized")
pca.fit(train_x_np)

def transform(x):
    x_np = x.detach().cpu().numpy()
    x_reduced = pca.transform(x_np)
    return torch.tensor(x_reduced, dtype=torch.float32)

train_x_red = transform(train_x)
valid_x_red = transform(valid_x)
test_x_red = transform(test_x)

train_new = [(train_x_red, train_y, train_w, train_group, train_instance)]
valid_new = [(valid_x_red, valid_y, valid_w, valid_group, valid_instance)]
test_new = [(test_x_red, test_y, test_w, test_group, test_instance)]

with open("/lustre/nj/cvpr2026/pickles/pca/vindr-msl-8.pkl", "wb") as f:
    pickle.dump((train_new, valid_new, test_new), f)
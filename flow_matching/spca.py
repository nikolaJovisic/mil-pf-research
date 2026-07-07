import pickle
import torch
import numpy as np
import sys
sys.path.append("../head_training")

pkl_path = "/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl"

alpha = 1.0
n_components = 128

with open(pkl_path, "rb") as f:
    train_ds, valid_ds, test_ds = pickle.load(f)

train_x, train_y, train_w, train_group, train_instance = train_ds[0]
valid_x, valid_y, valid_w, valid_group, valid_instance = valid_ds[0]
test_x, test_y, test_w, test_group, test_instance = test_ds[0]

train_x_np = train_x.detach().cpu().numpy()
train_y_np = train_y[train_group].detach().cpu().numpy().astype(np.float64)

train_y_np = (train_y_np - train_y_np.mean()) / (train_y_np.std() + 1e-8)

X = train_x_np - train_x_np.mean(axis=0, keepdims=True)
y = train_y_np.reshape(-1, 1)

cov_term = X.T @ X
sup_term = X.T @ y @ y.T @ X

M = cov_term + alpha * sup_term

eigvals, eigvecs = np.linalg.eigh(M)
idx = np.argsort(eigvals)[::-1]
W = eigvecs[:, idx[:n_components]]

def transform(x):
    x_np = x.detach().cpu().numpy()
    x_np = x_np - train_x_np.mean(axis=0, keepdims=True)
    x_red = x_np @ W
    return torch.tensor(x_red, dtype=torch.float32)

train_x_red = transform(train_x)
valid_x_red = transform(valid_x)
test_x_red = transform(test_x)

train_new = [(train_x_red, train_y, train_w, train_group, train_instance)]
valid_new = [(valid_x_red, valid_y, valid_w, valid_group, valid_instance)]
test_new = [(test_x_red, test_y, test_w, test_group, test_instance)]

with open("/lustre/nj/cvpr2026/pickles/spca/medsiglip-inf-128.pkl", "wb") as f:
    pickle.dump((train_new, valid_new, test_new), f)

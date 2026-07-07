import pickle
import torch
import sys
from einops import rearrange


def subsample_groups(input_path, output_path, fraction=0.2, seed=42):
    torch.manual_seed(seed)

    train_ds, val_ds, test_ds = pickle.load(open(input_path, "rb"))
    x, y, w, group, instance_type = train_ds[0]

    y = rearrange(y, "g 1 -> g")
    w = rearrange(w, "g 1 -> g")

    num_groups = y.shape[0]
    all_group_ids = torch.arange(num_groups, device=y.device)

    selected_groups = []

    classes = torch.unique(y)

    for c in classes:
        class_group_ids = all_group_ids[y == c]
        n_keep = max(1, int(len(class_group_ids) * fraction))

        perm = torch.randperm(len(class_group_ids), device=y.device)
        keep = class_group_ids[perm[:n_keep]]

        selected_groups.append(keep)

    selected_groups = torch.cat(selected_groups)

    instance_mask = torch.isin(group, selected_groups)

    x_new = x[instance_mask]
    group_new = group[instance_mask]
    instance_type_new = instance_type[instance_mask]

    y_new = y[selected_groups]
    w_new = w[selected_groups]

    _, group_new = torch.unique(group_new, sorted=True, return_inverse=True) 

    y_new = rearrange(y_new, "g -> g 1")
    w_new = rearrange(w_new, "g -> g 1")

    new_train_ds = [(x_new, y_new, w_new, group_new, instance_type_new)]

    pickle.dump((new_train_ds, val_ds, test_ds), open(output_path, "wb"))

    print("Original groups:", num_groups)
    print("New groups:", len(selected_groups))
    print("Original instances:", x.shape[0])
    print("New instances:", x_new.shape[0])


if __name__ == "__main__":
    input_pickle = '/lustre/nj/cvpr2026/pickles/pca/medsiglip-inf-128.pkl'
    output_pickle = '/lustre/nj/cvpr2026/pickles/pca/medsiglip-inf-128-subset.pkl'

    subsample_groups(input_pickle, output_pickle, fraction=0.5)
import pickle
import torch
import numpy as np

from utils import (
    compute_fid,
    compute_internal_fid,
    compute_vs_noise_fid,
    compute_interinstance_fid,
    compute_interlabel_fid
)

pkl_path = "/lustre/nj/cvpr2026/pickles/pca/vindr-msl-128.pkl"

with open(pkl_path, "rb") as f:
    train_ds, valid_ds, test_ds = pickle.load(f)

train_x, train_y_group, train_w, train_group, train_instance = train_ds[0]

train_y = train_y_group.squeeze()

num_groups_per_class = 2000

fids_noise = []
fids_halves = []
fids_instance = []
fids_interlabel = []

for i in range(10):
    pos_groups = torch.where(train_y == 1)[0]
    neg_groups = torch.where(train_y == 0)[0]

    pos_sampled = pos_groups[torch.randperm(len(pos_groups))[:num_groups_per_class]]
    neg_sampled = neg_groups[torch.randperm(len(neg_groups))[:num_groups_per_class]]

    selected_groups = torch.cat([pos_sampled, neg_sampled], dim=0)

    mask = torch.isin(train_group, selected_groups)

    x_subset = train_x[mask]
    group_subset = train_group[mask]
    instance_subset = train_instance[mask]

    y_subset = train_y[group_subset]

    fid_half = compute_internal_fid(x_subset)
    fids_halves.append(fid_half)

    fid_noise = compute_vs_noise_fid(x_subset)
    fids_noise.append(fid_noise)

    fid_instance = compute_interinstance_fid(
        x_subset,
        instance_subset
    )
    fids_instance.append(fid_instance)

    fid_interlabel = compute_interlabel_fid(
        x_subset,
        instance_subset,
        y_subset,
        instance_type=0
    )
    fids_interlabel.append(fid_interlabel)

    print(
        f"Run {i+1}: "
        f"FID(internal) = {fid_half}, "
        f"FID(vs noise) = {fid_noise}, "
        f"FID(interinstance) = {fid_instance}, "
        f"FID(interlabel (gl stream)) = {fid_interlabel}"
    )

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from icecream import ic

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
sys.path.append("../head_training")


orig_pkl = "/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl"
synth_pkl = "/lustre/nj/cvpr2026/pickles/setflow/medsiglip-inf-synth-only.pkl"
out_dir = "lda_figs"
os.makedirs(out_dir, exist_ok=True)

def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def plot(noise_scale=1.0, filename="plot.png", dpi=300, s=5, alpha=0.4, **arrays):
    plt.figure(figsize=(8, 6))
    for name, x in arrays.items():
        x = np.asarray(x).reshape(-1, 1)
        noise = np.random.uniform(-noise_scale, noise_scale, size=(x.shape[0], 1))
        points = np.hstack([x, noise])
        plt.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, label=name)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(filename), dpi=dpi)
    plt.close()

def train_lda(x: torch.Tensor, y: torch.Tensor, n_components=None, solver="svd"):
    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver=solver
    )
    lda.fit(x, y)
    return lda

def subsample(z, mask, n):
    rng = np.random.default_rng(seed=42)
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return z[idx]

def preprocess():
    orig_data, _, _ = load_data(orig_pkl)
    synth_data = load_data(synth_pkl)

    x_orig, y_orig, _, group_orig, it_orig = orig_data[0]
    x_synth, y_synth, _, group_synth, it_synth = synth_data[0]

    y_orig = rearrange(y_orig, 'b 1 -> b')
    y_synth = rearrange(y_synth, 'b 1 -> b')

    group_synth -= group_synth.min()

    x_orig = x_orig.cpu().numpy()
    x_synth = x_synth.cpu().numpy()
    y_orig = y_orig[group_orig].cpu().numpy()
    y_synth = y_synth[group_synth].cpu().numpy()
    it_orig = it_orig.cpu().numpy()
    it_synth = it_synth.cpu().numpy()
    return x_orig, y_orig, it_orig, x_synth, y_synth, it_synth

def train(instance_type):
    x_orig, y_orig, it_orig, x_synth, y_synth, it_synth = preprocess()

    x_orig = x_orig[it_orig == instance_type]
    y_orig = y_orig[it_orig == instance_type]

    lda = train_lda(x_orig, y_orig)
    pickle.dump(lda, open(os.path.join(out_dir, f"lda_{instance_type}_model.pkl"), "wb"))

def visualize(instance_type):
    x_orig, y_orig, it_orig, x_synth, y_synth, it_synth = preprocess()

    x_orig = x_orig[it_orig == instance_type]
    y_orig = y_orig[it_orig == instance_type]
    x_synth = x_synth[it_synth == instance_type]
    y_synth = y_synth[it_synth == instance_type]

    lda = pickle.load(open(os.path.join(out_dir, f"lda_{instance_type}_model.pkl"), "rb"))

    z_orig = lda.transform(x_orig)
    z_synth = lda.transform(x_synth)

    orig_pos = (y_orig == 1)
    orig_neg = (y_orig == 0)
    synth_pos = (y_synth == 1)
    synth_neg = (y_synth == 0)

    n_vis = min(synth_pos.sum(), synth_neg.sum())

    z_orig_pos = subsample(z_orig, orig_pos, n_vis)
    z_orig_neg = subsample(z_orig, orig_neg, n_vis)
    z_synth_pos = subsample(z_synth, synth_pos, n_vis)
    z_synth_neg = subsample(z_synth, synth_neg, n_vis)

    plot(orig_pos=z_orig_pos, orig_neg=z_orig_neg, filename=f"{out_dir}/orig_{instance_type}.png")
    plot(synth_pos=z_synth_pos, synth_neg=z_synth_neg, filename=f"{out_dir}/syn_{instance_type}.png")

def monitor(iteration, x, y, group, instance_type):
    for it in [0, 1]:
        mask = (instance_type == it)
        x_transformed = x[mask].cpu().numpy()
        y_transformed = y[mask].cpu().numpy()
        lda = pickle.load(open(os.path.join(out_dir, f"lda_{it}_model.pkl"), "rb"))
        z = lda.transform(x_transformed)
        pos = (y_transformed == 1)
        neg = (y_transformed == 0)
        plot(pos=z[pos], neg=z[neg], filename=f"monitoring/monitor_it{it}_iter{iteration}.png")


monitor('$test', torch.randn(100, 1152), torch.randint(0, 2, (100,)), torch.zeros(100), torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)]))


# train(instance_type=0)
# train(instance_type=1)
# visualize(instance_type=0)
# visualize(instance_type=1)
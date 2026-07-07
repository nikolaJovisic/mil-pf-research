import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.linalg import sqrtm
from icecream import ic

def sample_alpha(batch_size, device):
    alpha = torch.normal(
        mean=0.5,
        std=0.15,
        size=(batch_size, 1),
        device=device
    )
    return alpha.clamp(0.2, 0.8)

def plot_lda(
    x,
    y,
    title="lda_plot",
    jitter_scale=0.05,
    seed=0,
):
    rng = np.random.default_rng(seed)

    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    y_np = y_np.reshape(-1)

    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(x_np, y_np)[:, 0]

    jitter = rng.normal(0.0, jitter_scale, size=len(z))

    plt.figure(figsize=(6, 5))
    for c in np.unique(y_np):
        mask = y_np == c
        plt.scatter(
            z[mask],
            jitter[mask],
            s=8,
            alpha=0.7,
            label=f"class {c}",
        )

    plt.yticks([])
    plt.xlabel("LDA direction")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.png")

    return lda


def plot_pca(
    x,
    y,
    title="pca_plot",
):
    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    y_np = y_np.reshape(-1)

    pca = PCA(n_components=2)
    z = pca.fit_transform(x_np)

    plt.figure(figsize=(6, 5))
    for c in np.unique(y_np):
        mask = y_np == c
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=8,
            alpha=0.7,
            label=f"class {c}",
        )

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.png")

    return pca

def fit_pca_lda(x, y):
    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    y_np = y_np.reshape(-1)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(x_np, y_np)

    pca = PCA(n_components=2)
    pca.fit(x_np)

    return pca, lda


def init_pca_lda_figure(
    x,
    y,
    pca,
    lda,
    title="LDA (left) and PCA (right)",
    jitter_scale=0.05,
    seed=0,
):
    rng = np.random.default_rng(seed)

    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    y_np = y_np.reshape(-1)

    z_lda = lda.transform(x_np)[:, 0]
    jitter = rng.normal(0.0, jitter_scale, size=len(z_lda))

    z_pca = pca.transform(x_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for c in np.unique(y_np):
        m = y_np == c
        color = "orange" if c == 0 else "blue"
        axes[0].scatter(z_lda[m], jitter[m], s=8, alpha=0.6, color=color, label=f"class {c}")
        axes[1].scatter(z_pca[m, 0], z_pca[m, 1], s=8, alpha=0.6, color=color, label=f"class {c}")

    axes[0].set_yticks([])
    axes[0].set_xlabel("LDA direction")
    axes[0].set_title("LDA (supervised)")

    evr = pca.explained_variance_ratio_
    axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axes[1].set_title("PCA (unsupervised)")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig, axes

def add_synth_and_save(
    fig,
    axes,
    x_synth,
    y_synth,
    pca,
    lda,
    jitter_scale=0.05,
    seed=0,
    save_path=None,
):
    rng = np.random.default_rng(seed)

    x = x_synth.detach().cpu().numpy() if hasattr(x_synth, "detach") else np.asarray(x_synth)
    y = y_synth.detach().cpu().numpy() if hasattr(y_synth, "detach") else np.asarray(y_synth)
    y = y.reshape(-1)

    z_lda = lda.transform(x)[:, 0]
    jitter = rng.normal(0.0, jitter_scale, size=len(z_lda))

    z_pca = pca.transform(x)

    s = []
    for c in np.unique(y):
        m = y == c
        color = "magenta" if c == 0 else "cyan"
        s.append(
            axes[0].scatter(
                z_lda[m],
                jitter[m],
                s=14,
                marker="x",
                alpha=0.85,
                color=color,
                label=f"synth {c}",
            )
        )
        s.append(
            axes[1].scatter(
                z_pca[m, 0],
                z_pca[m, 1],
                s=14,
                marker="x",
                alpha=0.85,
                color=color,
                label=f"synth {c}",
            )
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

    for artist in s:
        artist.remove()


def compute_fm_dataset_stats(y, instance_type):
    _, class_counts = torch.unique(y, return_counts=True)

    changes = instance_type[1:] != instance_type[:-1]
    indices = torch.cat([
        torch.tensor([0], device=instance_type.device),
        torch.nonzero(changes, as_tuple=False).flatten() + 1,
        torch.tensor([instance_type.numel()], device=instance_type.device)
    ])

    run_lengths = indices[1:] - indices[:-1]
    run_values = instance_type[indices[:-1]]

    ones_runs = run_lengths[run_values == 1]
    zeros_runs = run_lengths[run_values == 0]

    mean_ones = ones_runs.float().mean()
    std_ones = ones_runs.float().std(unbiased=False)

    mean_zeros = zeros_runs.float().mean()
    std_zeros = zeros_runs.float().std(unbiased=False)
    #ic(class_counts, mean_ones, std_ones, mean_zeros, std_zeros)

def compute_fid_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def compute_fid(x1, x2, eps=1e-6):
    if torch.is_tensor(x1):
        x1 = x1.detach().cpu().numpy()
    if torch.is_tensor(x2):
        x2 = x2.detach().cpu().numpy()
    mu1, sigma1 = compute_fid_stats(x1)
    mu2, sigma2 = compute_fid_stats(x2)
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_internal_fid(x, eps=1e-6):
    """
    FID between two random non-overlapping halves
    of the same dataset.
    """
    if torch.is_tensor(x):
        x = x.detach()

    N = x.shape[0]
    half = N // 2

    perm = torch.randperm(N)
    idx1 = perm[:half]
    idx2 = perm[half:2 * half]

    x1 = x[idx1]
    x2 = x[idx2]

    return compute_fid(x1, x2, eps=eps)


def compute_vs_noise_fid(x, eps=1e-6):
    """
    FID between a random half of the dataset
    and Gaussian noise with matching dimensionality.
    """
    if torch.is_tensor(x):
        x = x.detach()

    N, D = x.shape
    half = N // 2

    perm = torch.randperm(N)
    idx = perm[:half]

    x_half = x[idx]
    noise = torch.randn(half, D, device=x_half.device)

    return compute_fid(x_half, noise, eps=eps)


def compute_interinstance_fid(x, instance_labels, eps=1e-6): # 0.42 for originals
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(instance_labels):
        instance_labels = instance_labels.detach().cpu().numpy()

    instance_labels = np.asarray(instance_labels)

    idx_0 = np.where(instance_labels == 0)[0]
    idx_1 = np.where(instance_labels == 1)[0]

    n0 = len(idx_0)

    if n0 == 0:
        raise ValueError("No samples with instance_type=0.")
    if len(idx_1) < n0:
        raise ValueError("Not enough instance_type=1 samples to match.")

    sampled_1 = np.random.choice(idx_1, size=n0, replace=False)

    x0 = x[idx_0]
    x1 = x[sampled_1]

    return compute_fid(
        torch.from_numpy(x0),
        torch.from_numpy(x1),
        eps=eps
    )

def compute_interlabel_fid(x, instance_labels, y_labels, instance_type=0, eps=1e-6): # 0.006 for originals
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(instance_labels):
        instance_labels = instance_labels.detach().cpu().numpy()
    if torch.is_tensor(y_labels):
        y_labels = y_labels.detach().cpu().numpy()

    mask_instance = (instance_labels == instance_type)

    idx_instance = np.where(mask_instance)[0]

    y_instance = y_labels[idx_instance]

    idx_y1 = idx_instance[y_instance == 1]
    idx_y0 = idx_instance[y_instance == 0]

    n = min(len(idx_y1), len(idx_y0))

    sampled_y0 = np.random.choice(idx_y0, size=n, replace=False)

    x_y1 = x[idx_y1]
    x_y0 = x[sampled_y0]

    return compute_fid(x_y1, x_y0, eps=eps)

import torch


def nn_subset_monitor(
    x_real,
    y_real,
    group_real,
    instance_type_real,
    x_syn,
    y_syn,
    group_syn,
    instance_type_syn,
    target_instance_type=0,
    target_label=1,
):
    """
    Computes internal and cross nearest-neighbour distances
    on a filtered subset defined by:
        instance_type == target_instance_type
        label == target_label

    Returns a dict with means and stds.
    """
    x_real = x_real.cpu()
    y_real = y_real.cpu()
    group_real = group_real.cpu()
    instance_type_real = instance_type_real.cpu()

    def filter_subset(x, y, instance_type):
        mask = (instance_type == target_instance_type) & (
            y.squeeze() == target_label
        )
        return x[mask]

    def nn_internal(x):
        if x.shape[0] < 2:
            return None
        dists = torch.cdist(x, x)
        dists.fill_diagonal_(float("inf"))
        return dists.min(dim=1).values

    def nn_cross(x_a, x_b):
        if x_a.shape[0] == 0 or x_b.shape[0] == 0:
            return None
        dists = torch.cdist(x_a, x_b)
        return dists.min(dim=1).values

    x_real_sub = filter_subset(x_real, y_real, instance_type_real)
    x_syn_sub = filter_subset(x_syn, y_syn, instance_type_syn)[:x_real_sub.shape[0]]

    real_internal = nn_internal(x_real_sub)
    syn_internal = nn_internal(x_syn_sub)
    real_to_syn = nn_cross(x_real_sub, x_syn_sub)
    syn_to_real = nn_cross(x_syn_sub, x_real_sub)

    def summarize(t):
        if t is None:
            return {"mean": None, "std": None}
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
        }

    return {
        "count_real": x_real_sub.shape[0],
        "internal_real": summarize(real_internal),
        "internal_syn": summarize(syn_internal),
        "real_to_syn": summarize(real_to_syn),
        "syn_to_real": summarize(syn_to_real),
    }

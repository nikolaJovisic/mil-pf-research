import pickle
import torch

input_pkl = "/lustre/nj/cvpr2026/pickles/pca/medsiglip-inf-128.pkl"
synthetic_pkl = "/lustre/nj/cvpr2026/pickles/setflow/msl-100p-128-synth-only.pkl"


def filter_subset(x, y, instance_type):
    mask = (instance_type == 0) & (y.squeeze() == 1)
    return x[mask]

def nn_internal(x):
    dists = torch.cdist(x, x)
    dists.fill_diagonal_(float("inf"))
    return dists.min(dim=1).values

def nn_cross(x_a, x_b):
    dists = torch.cdist(x_a, x_b)
    return dists.min(dim=1).values

def summarize(name, tensor):
    print(
        f"{name}: mean={tensor.mean().item():.6f}, "
        f"std={tensor.std().item():.6f}"
    )

def main():
    train_ds, _, _ = pickle.load(open(input_pkl, "rb"))
    train_syn = pickle.load(open(synthetic_pkl, "rb"))

    x_real, y_real, _, group_real, instance_type_real = train_ds[0]
    x_syn, y_syn, _, group_syn, instance_type_syn = train_syn[0]
    group_syn = group_syn - group_syn.min()

    y_real = y_real[group_real]
    y_syn = y_syn[group_syn]

    x_real_sub = filter_subset(x_real, y_real, instance_type_real)
    x_syn_sub = filter_subset(x_syn, y_syn, instance_type_syn)

    print("Counts:")
    print("Original:", x_real_sub.shape[0])
    print("Synthetic:", x_syn_sub.shape[0])

    real_internal = nn_internal(x_real_sub)
    syn_internal = nn_internal(x_syn_sub)

    real_to_syn = nn_cross(x_real_sub, x_syn_sub)
    syn_to_real = nn_cross(x_syn_sub, x_real_sub)

    print("\nNearest neighbour distances:")
    summarize("Internal Original", real_internal)
    summarize("Internal Synthetic", syn_internal)
    summarize("Original -> Synthetic", real_to_syn)
    summarize("Synthetic -> Original", syn_to_real)


if __name__ == "__main__":
    main()

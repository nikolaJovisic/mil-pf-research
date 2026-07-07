import pickle
import torch
import numpy as np
import sys
import os

# -----------------------------
# Config
# -----------------------------
pkl_path = "/lustre/nj/cvpr2026/pickles/pca/medsiglip-inf-128.pkl"
output_dir = "./spectral_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
sys.path.append("../head_training")

with open(pkl_path, "rb") as f:
    train_ds, _, _ = pickle.load(f)

x, y, _, group, instance_type = train_ds[0]

# Use float64 for numerical stability
x = x.double()

N, D = x.shape
print(f"Data shape: {N} x {D}")

# -----------------------------
# Covariance
# -----------------------------
cov = (x.T @ x) / (N - 1)

# -----------------------------
# Eigen-decomposition
# -----------------------------
eigenvalues, eigenvectors = torch.linalg.eigh(cov)

# Sort descending (optional but cleaner)
idx = torch.argsort(eigenvalues, descending=True)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigen decomposition complete")

# -----------------------------
# Singular values
# -----------------------------
# Clamp tiny negatives due to numerical noise
eps = 1e-10
eigenvalues_clamped = torch.clamp(eigenvalues, min=eps)

singular_values = torch.sqrt(eigenvalues_clamped * (N - 1))

print("Singular values computed")

# -----------------------------
# Whitening
# -----------------------------
whitening_matrix = eigenvectors @ torch.diag(
    1.0 / torch.sqrt(eigenvalues_clamped)
)

X_white = x @ whitening_matrix

print("Whitening complete")

# -----------------------------
# Kurtosis (Fisher definition)
# -----------------------------
kurtosis = torch.mean(X_white ** 4, dim=0) - 3
avg_kurtosis = torch.mean(kurtosis)

print("Kurtosis computed")
print("Average kurtosis:", avg_kurtosis.item())

# -----------------------------
# Save to CSV
# -----------------------------
np.savetxt(
    os.path.join(output_dir, "eigenvalues.csv"),
    eigenvalues.cpu().numpy(),
    delimiter=",",
    fmt="%.12e"
)

np.savetxt(
    os.path.join(output_dir, "singular_values.csv"),
    singular_values.cpu().numpy(),
    delimiter=",",
    fmt="%.12e"
)

np.savetxt(
    os.path.join(output_dir, "kurtosis.csv"),
    kurtosis.cpu().numpy(),
    delimiter=",",
    fmt="%.12e"
)

print("Saved:")
print(f"- {output_dir}/eigenvalues.csv")
print(f"- {output_dir}/singular_values.csv")
print(f"- {output_dir}/kurtosis.csv")

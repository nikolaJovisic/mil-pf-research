from embeddings_dataset import EmbeddingsDataset
import torch
import pickle

def compute_embedding_eig_stats(dataset: EmbeddingsDataset):
    all_embeddings = []
    for batch in dataset:
        all_embeddings.append(batch[0])
    X = torch.cat(all_embeddings, dim=0)  
    
    # Normalize for stability (optional)
    X = X - X.mean(dim=0, keepdim=True)
    
    # Covariance matrix
    cov = (X.T @ X) / (X.shape[0] - 1)
    
    # Eigen decomposition
    eigvals = torch.linalg.eigvalsh(cov).real
    
    mean_eig = eigvals.mean().item()
    var_eig = eigvals.var(unbiased=False).item()
    
    return mean_eig, var_eig

baseline_datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/vitb-explora-baseline.pkl', 'rb'))
explora_datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/explora-v2-training_30000.pkl', 'rb'))

baseline_train_ds, baseline_valid_ds, baseline_test_ds = baseline_datasets
explora_train_ds, explora_valid_ds, explora_test_ds = explora_datasets

mean_eig_baseline, var_eig_baseline = compute_embedding_eig_stats(baseline_valid_ds)
mean_eig_explora, var_eig_explora = compute_embedding_eig_stats(explora_valid_ds)
print(f"Baseline Train Embeddings - Mean Eigenvalue: {mean_eig_baseline}, Variance of Eigenvalues: {var_eig_baseline}")
print(f"Explora Train Embeddings - Mean Eigenvalue: {mean_eig_explora}, Variance of Eigenvalues: {var_eig_explora}")

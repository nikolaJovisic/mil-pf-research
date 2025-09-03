import os
import torch
from torch.utils.data import Dataset

class PrecollatedDataset(Dataset):
    def __init__(self, pt_dir, device="cuda"):
        self.pt_dir = pt_dir
        self.device = device
        self.batch_files = sorted(
            [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
        )
        if not self.batch_files:
            raise ValueError(f"No .pt batches found in {pt_dir}")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        path = self.batch_files[idx]
        return torch.load(path, map_location=self.device)

import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize 
import h5py
from tqdm import tqdm
from einops import rearrange    
from icecream import ic
from collections import Counter
from utils.collate import collate
import psutil

class EmbeddingsDataset(Dataset):
    def __init__(self, h5_path, pos_labels, neg_labels, batch_size, pos_weight=1.0):
        self.pos_labels = set(pos_labels)
        self.neg_labels = set(neg_labels)

        print(f'Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB')
        with h5py.File(
            h5_path,
            "r",
            rdcc_nbytes= 50 * 1024 ** 3,
            libver="latest",
        ) as file:
        #with h5py.File(h5_path, 'r', driver='core', backing_store=False) as file:
            # labels = [int(group['label'][()]) for group in file.values()]
            ic(file['0']['embeddings'][()].shape)

            # counter = Counter(labels)
            # ic(counter)

            # unexpected = set(counter) - (self.pos_labels | self.neg_labels)
            # if unexpected:
            #     ic(f"Warning: unexpected labels {unexpected}")

            # self._compute_weights(counter, pos_weight)
            # ic(self.weights)
            self.weights = {0: 1, 1: 1}

            items = []
            for i, group in enumerate(tqdm(file.values(), desc="Loading embeddings")):
                if i % 200 == 0: 
                    print(f'Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB')
                label = int(group['label'][()])
                if label not in self.pos_labels | self.neg_labels:
                    continue
                label = int(label in self.pos_labels)
                embeddings = torch.from_numpy(group['embeddings'][()])
                weight = torch.tensor([self.weights[label]], dtype=torch.float32)
                label_tensor = torch.tensor([label], dtype=torch.float32)
                items.append((embeddings, label_tensor, weight))
            ic(len(items))

        self.batches = [batch for batch in tqdm(collate(items, batch_size))] 
        ic(len(self.batches))

    def _compute_weights(self, counter, pos_weight):
        pos_count = sum(counter[l] for l in self.pos_labels)
        neg_count = sum(counter[l] for l in self.neg_labels)
        ic(pos_count, neg_count)
        total = pos_count + neg_count
        self.weights = {0: (total / (2 * neg_count)),
                        1: pos_weight * (total / (2 * pos_count))}

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x_batch = self.batches[idx][0]
        ic(x_batch.numel() * x_batch.element_size())
        return self.batches[idx]

if __name__ == "__main__":
    EmbeddingsDataset(
        '/lustre/nj/dinov3-embeddings/dinov3-s-512-embed-light/train/embeddings.h5',
        pos_labels=[4, 5, 6],
        neg_labels=[1],
        batch_size=2048,
        pos_weight=1.0
    )
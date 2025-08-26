import torch
from torch.utils.data import IterableDataset
from torch.nn.functional import normalize 
import h5py
from einops import rearrange    
from icecream import ic
from collections import Counter

class EmbeddingsDataset(IterableDataset):
    def __init__(self, h5_path, pos_labels, neg_labels, pos_weight=1.0):
        self.path = h5_path
        self.pos_labels = set(pos_labels)
        self.neg_labels = set(neg_labels)

        with h5py.File(self.path, 'r') as file:
            labels = [int(group['label'][()]) for group in file.values()]
            ic(file['0']['images'][()].shape)

        counter = Counter(labels)
        ic(counter, pos_count, neg_count)

        unexpected = set(counter) - (self.pos_labels | self.neg_labels)
        if unexpected: ic(f"Warning: unexpected labels {unexpected}")

        self._compute_weights(pos_weight)
        ic(self.weights)

    def _compute_weights(self, pos_weight):
        pos_count = sum(counter[l] for l in self.pos_labels)
        neg_count = sum(counter[l] for l in self.neg_labels)
        total = pos_count + neg_count
        self.weights = {0: (total / (2 * neg_count)), 1: pos_weight * (total / (2 * pos_count))}

    def __iter__(self):
        with h5py.File(self.path, 'r') as file:
            for group in file.values():
                label = int(group['label'][()])
                if label not in self.pos_labels | self.neg_labels: continue
                label = int(label in self.pos_labels)
                embeddings = group['images'][()]
                embeddings = torch.from_numpy(embeddings)
                embeddings = normalize(embeddings, p=2, dim=1)

                weight = torch.tensor([self.weights[label]], dtype=torch.float32)
                label  = torch.tensor([label], dtype=torch.float32)

                yield embeddings, label, weight


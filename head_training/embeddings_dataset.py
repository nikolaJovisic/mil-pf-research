import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize 
import h5py
from einops import rearrange    
from icecream import ic
from collections import Counter
from utils.collate import collate

class EmbeddingsDataset(Dataset):
    def __init__(self, h5_path, pos_labels, neg_labels, batch_size, pos_weight=1.0):
        self.pos_labels = set(pos_labels)
        self.neg_labels = set(neg_labels)

        with h5py.File(h5_path, 'r', driver='core', backing_store=False) as file:
            labels = [int(group['label'][()]) for group in file.values()]
            ic(file['0']['embeddings'][()].shape)

            counter = Counter(labels)
            ic(counter)

            unexpected = set(counter) - (self.pos_labels | self.neg_labels)
            if unexpected:
                ic(f"Warning: unexpected labels {unexpected}")

            self._compute_weights(counter, pos_weight)
            ic(self.weights)

            items = []
            for group in file.values():
                label = int(group['label'][()])
                if label not in self.pos_labels | self.neg_labels:
                    continue
                label = int(label in self.pos_labels)
                embeddings = torch.from_numpy(group['embeddings'][()])
                weight = torch.tensor([self.weights[label]], dtype=torch.float32)
                label_tensor = torch.tensor([label], dtype=torch.float32)
                items.append((embeddings, label_tensor, weight))
            ic(len(items))

        self.batches = [batch for batch in collate(items, batch_size)] 
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
        return self.batches[idx]

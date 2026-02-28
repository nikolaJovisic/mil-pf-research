import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize 
import h5py
from tqdm import tqdm
from einops import rearrange    
from icecream import ic
from collections import Counter
from head_training.utils.collate import collate
import psutil
from head_training.utils.flatten_group import FlattenGroup

class EmbeddingsDataset(Dataset):
    def __init__(self, h5_paths, pos_labels, neg_labels, batch_size, flatten=False, pos_weight=1.0):
        self.pos_labels = set(pos_labels)
        self.neg_labels = set(neg_labels)
        self.flatten = flatten


        buffs = []
        for h5_path in h5_paths:
            with open(h5_path, "rb") as _f:
                buffs.append(_f.read())
        
        labels = []
        for _buf in buffs:
            with h5py.File.in_memory(file_image=_buf) as file: 
                labels.extend([int(group['label'][()]) for group in file.values()])

        counter = Counter(labels)
        ic(counter)

        unexpected = set(counter) - (self.pos_labels | self.neg_labels)
        if unexpected:
            ic(f"Warning: unexpected labels {unexpected}")

        self._compute_weights(counter, pos_weight)
        ic(self.weights)
        
        items = []
        for _buf in buffs:
            with h5py.File.in_memory(file_image=_buf) as file:
                for i, group in enumerate(tqdm(file.values(), desc="Iterating embeddings.")):
                    label = int(group['label'][()])
                    if label not in self.pos_labels | self.neg_labels:
                        continue
                    label = int(label in self.pos_labels)
                    images = torch.from_numpy(group['images'][()])
                    tiles = torch.from_numpy(group['tiles'][()]) if 'tiles' in group else torch.empty((0, images.shape[1]))
                    weight = torch.tensor([self.weights[label]], dtype=torch.float32)
                    label_tensor = torch.tensor([label], dtype=torch.float32)
                    all_images = torch.cat([images, tiles], dim=0)
                    instance_type = torch.cat([torch.full((images.shape[0],), 0, dtype=torch.long), torch.full((tiles.shape[0],), 1, dtype=torch.long)], dim=0)
                    items.append((all_images, label_tensor, weight, instance_type))  
                ic(len(items))
            
        if self.flatten:
            items = FlattenGroup(items)

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
        return self.batches[idx]

if __name__ == "__main__":
    EmbeddingsDataset(
        '/lustre/nj/dinov3-embeddings/dinov3-s-512-embed-light/train/embeddings.h5',
        pos_labels=[4, 5, 6],
        neg_labels=[1],
        batch_size=2048,
        pos_weight=1.0
    )
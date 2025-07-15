import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from collections import defaultdict, Counter

class EmbeddingsDataset(Dataset):
    """
    Example dataset_config:

    dataset_config = {
        'data/file1.h5': {'pos': [1, 3], 'neg': [0]},
        'data/file2.h5': {'pos': [5], 'neg': [2, 4]},
    }
    """
    def __init__(self, dataset_config, use_tiles=False, pos_weight=1.0):
        self.dataset_config = dataset_config
        self.use_tiles = use_tiles
        self.embeddings = []
        self.labels = []
        self.sample_weights = []
        self.group_indices = []
        self.instance_types = []
        self.class_counts = defaultdict(int)
        self.dataset_class_counts = {}
        self.dataset_raw_counts = {}
        self.total_raw_counts = Counter()

        group_start_idx = 0

        for path, label_mapping in dataset_config.items():
            pos_labels = label_mapping.get('pos', [])
            neg_labels = label_mapping.get('neg', [])

            local_counts = defaultdict(int)
            local_raw_counts = Counter()

            with h5py.File(path, 'r') as f:
                for group_name in f:
                    group = f[group_name]
                    embs = group['images'][()]
                    types = np.zeros((embs.shape[0],), dtype=np.int64)

                    if self.use_tiles and 'tiles' in group:
                        tile_embs = group['tiles'][()]
                        tile_types = np.ones((tile_embs.shape[0],), dtype=np.int64)
                        embs = np.concatenate([embs, tile_embs], axis=0)
                        types = np.concatenate([types, tile_types], axis=0)

                    raw_label = int(group['label'][()])
                    local_raw_counts[raw_label] += embs.shape[0]
                    self.total_raw_counts[raw_label] += embs.shape[0]

                    if raw_label in pos_labels:
                        label = 1
                    elif raw_label in neg_labels:
                        label = 0
                    else:
                        continue

                    self.embeddings.append(embs)
                    self.instance_types.append(types)
                    self.labels.append(label)
                    self.sample_weights.append(embs.shape[0])
                    self.group_indices.append((group_start_idx, group_start_idx + embs.shape[0]))
                    group_start_idx += embs.shape[0]
                    self.class_counts[label] += embs.shape[0]
                    local_counts[label] += embs.shape[0]

            self.dataset_class_counts[path] = dict(local_counts)
            self.dataset_raw_counts[path] = dict(local_raw_counts)

        for path in dataset_config:
            pos = self.dataset_class_counts[path].get(1, 0)
            neg = self.dataset_class_counts[path].get(0, 0)
            print(f"{path} -> pos: {pos}, neg: {neg}")
            print(f"{path} -> raw label counts: {self.dataset_raw_counts[path]}")

        print(f"Total -> pos: {self.class_counts.get(1, 0)}, neg: {self.class_counts.get(0, 0)}")
        print(f"Total -> raw label counts: {dict(self.total_raw_counts)}")

        embs_np = np.concatenate(self.embeddings)
        embs_tensor = torch.tensor(embs_np, dtype=torch.float32)
        self.embeddings = torch.nn.functional.normalize(embs_tensor, p=2, dim=1)

        self.instance_types = torch.tensor(np.concatenate(self.instance_types), dtype=torch.int64)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
        self.sample_weights = self._compute_sample_weights(pos_weight)

    def _compute_sample_weights(self, positive_class_weight_multiplier=1.0, return_dict=False):
        total = sum(self.class_counts.values())
        weights = {
            cls: (total / (2 * count)) * (positive_class_weight_multiplier if cls == 1 else 1)
            for cls, count in self.class_counts.items()
        }
        if return_dict:
            return weights

        weight_list = [weights[int(lbl.item())] for lbl in self.labels]
        weights_tensor = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)
        return torch.clamp(weights_tensor, min=1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start, end = self.group_indices[idx]
        group = self.embeddings[start:end]
        instance_type = self.instance_types[start:end]

        if self.use_tiles and not (instance_type == 1).any():
            # Add one fake tile: 768-d zero vector
            zero_tile = torch.zeros((1, group.size(1)), dtype=group.dtype)
            group = torch.cat([group, zero_tile], dim=0)

            tile_type = torch.tensor([1], dtype=instance_type.dtype)
            instance_type = torch.cat([instance_type, tile_type], dim=0)

        return group, self.labels[idx], self.sample_weights[idx], instance_type

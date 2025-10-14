import os
import torch
import numpy as np
import threading
import queue
from torch.utils.data import IterableDataset, DataLoader

BASE_PATH = "/lustre/nj/cvpr2026/ds_prep"
WORLD_SIZE = 6

class ShardedMammoIterable(IterableDataset):
    def __init__(self, split: str, rank: int, tiles: bool, num_splits: int = None, prefetch: int = 2):
        self.split = split
        self.rank = rank
        self.tiles = tiles
        self.num_splits = num_splits
        self.prefetch = prefetch

        subdir = "tiles" if tiles else "images"
        self.data_dir = os.path.join(BASE_PATH, f"ws{WORLD_SIZE}", f"r{rank}", split, subdir)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.files = sorted(
            [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".pt")]
        )
        if not self.files:
            raise RuntimeError(f"No .pt files found in {self.data_dir}")

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch)

        def producer():
            for fpath in self.files:
                images, ids, labels = torch.load(fpath)

                if self.num_splits and self.num_splits > 1:
                    for sub in self._split_by_groups(images, ids, labels, self.num_splits):
                        q.put(sub)
                else:
                    q.put((images, ids, labels))
            q.put(None)  

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

    def _split_by_groups(self, images, ids, labels, num_splits):
        ids = np.array(ids)
        unique_ids, first_idx, counts = np.unique(ids, return_index=True, return_counts=True)

        group_ranges = [(start, start + c) for start, c in zip(first_idx, counts)]
        groups_per_split = max(1, len(group_ranges) // num_splits)

        for i in range(0, len(group_ranges), groups_per_split):
            part_ranges = group_ranges[i:i + groups_per_split]
            part_images, part_ids, part_labels = [], [], []
            for s, e in part_ranges:
                part_images.extend(images[s:e])
                part_ids.extend(ids[s:e])
                part_labels.extend(labels[s:e])
            yield part_images, part_ids, part_labels


def get_sharded_iter_dataloader(dataset):
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

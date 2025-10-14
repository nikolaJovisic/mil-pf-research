from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch
from enum import Enum, auto

from mammo_datasets import *

class BatchDataset(IterableDataset):
    def __init__(self, dataset, batch_size, tiles):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_samples = len(dataset)
        self.tiles = tiles

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_stride = 1
        else:
            iter_start = worker_info.id
            iter_stride = worker_info.num_workers

        batch_images = []
        batch_i = []
        batch_labels = []

        for i in range(iter_start, self.total_samples, iter_stride):
            if self.tiles:
                _, images, label = self.dataset[i]
            else:
                images, label = self.dataset[i]
            
            gid = i
            if hasattr(self.dataset, "indices"):
                gid = self.dataset.indices[i]

            batch_images.extend(images)
            batch_i.extend([gid] * len(images))
            batch_labels.extend([label] * len(images))

            while len(batch_images) >= self.batch_size:
                yield (
                    batch_images[:self.batch_size],
                    batch_i[:self.batch_size],
                    batch_labels[:self.batch_size]
                )
                batch_images = batch_images[self.batch_size:]
                batch_i = batch_i[self.batch_size:]
                batch_labels = batch_labels[self.batch_size:]

        if batch_images:
            yield batch_images, batch_i, batch_labels

def get_batched_dataloader(dataset, batch_size=8, num_workers=16, tiles=True):
    return DataLoader(BatchDataset(dataset, batch_size, tiles), batch_size=None, num_workers=num_workers)

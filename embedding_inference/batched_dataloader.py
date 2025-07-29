from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from enum import Enum, auto

from shim import *

class BatchEnum(Enum):
    IMAGE = auto()
    TILE = auto()

class BatchDataset(IterableDataset):
    def __init__(self, dataset, batch_enum, batch_size):
        
        if dataset.return_mode != ReturnMode.BREAST_TILES_LABEL:
            raise NotImplementedError("Cannot load this return mode in batches yet.")
        
        self.dataset = dataset
        self.batch_enum = batch_enum
        self.batch_size = batch_size
        self.total_samples = len(dataset)

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
            images, tiles, label = self.dataset[i]

            if self.batch_enum == BatchEnum.TILE:
                images = tiles
            
            batch_images.extend(images)
            batch_i.extend([i] * len(images))
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

def get_batched_dataloader(dataset, batch_enum=BatchEnum.IMAGE, batch_size=8, num_workers=16):
    return DataLoader(BatchDataset(dataset, batch_enum, batch_size), batch_size=None, num_workers=num_workers)

import torch
import os
from mammo_datasets import *
from torch.utils.data import Subset
from batched_dataloader import get_batched_dataloader

def save_dataset(dataset, out_dir, batch_size=4096, num_workers=8, tiles=False):
    os.makedirs(out_dir, exist_ok=True)
    loader = get_batched_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, tiles=tiles) 

    for batch_idx, batch in enumerate(loader):
        file_path = os.path.join(out_dir, f"{batch_idx:03d}.pt")
        torch.save(batch, file_path)
        print(f"Saved {file_path} ({len(batch)} samples)")

def shard_indices(n, world_size, rank):
    q, r = divmod(n, world_size)
    start = rank * q + min(rank, r)
    end = start + q + (1 if rank < r else 0)
    return list(range(start, end))

def save_split(split, world_size, rank):
    ds_params = {
        "dataset": DatasetEnum.EMBED,
        "labels": [1, 4, 5, 6],
        "convert_to": ConvertTo.RGB_TENSOR_IMGNET_NORM,
        "split": split,
        "tile_size": 518,
        "final_resize": 518,
    }

    ds_images_params = ds_params.copy()
    ds_images_params["return_mode"] = ReturnMode.BREAST_LABEL

    ds_tiles_params = ds_params.copy()
    ds_tiles_params["return_mode"] = ReturnMode.BREAST_TILES_LABEL

    ds_images_full = MammoDataset(**ds_images_params)
    ds_tiles_full = MammoDataset(**ds_tiles_params)

    idx = shard_indices(len(ds_images_full), world_size, rank)
    ds_images = Subset(ds_images_full, idx)
    ds_tiles = Subset(ds_tiles_full, idx)

    save_dataset(ds_tiles, f'/lustre/nj/cvpr2026/ds_prep/ws{world_size}/r{rank}/{split}/tiles', tiles=True)
    save_dataset(ds_images, f'/lustre/nj/cvpr2026/ds_prep/ws{world_size}/r{rank}/{split}/images', tiles=False)

if __name__ == "__main__":
    num_gpus = 6
    for split in ['train', 'valid', 'test']:
        for rank in range(num_gpus):
            save_split(split, num_gpus, rank)
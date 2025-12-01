import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
import h5py
from omegaconf import OmegaConf
from mammo_datasets import *
from embedding_inference_ import EmbeddingInference
from ds_loader import ShardedMammoIterable

def shard_indices(n, world_size, rank):
    q, r = divmod(n, world_size)
    start = rank * q + min(rank, r)
    end = start + q + (1 if rank < r else 0)
    return list(range(start, end))

def run_worker(rank, split, world_size):
    torch.cuda.set_device(rank)
    #limits = {'train': 2000, 'valid': 300, 'test': 500}
    ds_params = {
        "dataset": DatasetEnum.EMBED,
        "labels": [1, 4, 5, 6],
        "convert_to": ConvertTo.RGB_TENSOR,
        "split": split,
        "tile_size": 448,
        "final_resize": 448,
        "tile_overlap": 0
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

    # ds_images = ShardedMammoIterable(split, rank, tiles=False, num_splits=4, prefetch=256)
    # ds_tiles = ShardedMammoIterable(split, rank, tiles=True, num_splits=32)

    cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    cfg.run_name = f'{split}-gpu{rank}'
    embedding_inference = EmbeddingInference(ds_images, ds_tiles, cfg, device=f'cuda:{rank}')
    embedding_inference.run_images()
    embedding_inference.run_tiles()

def merge_split(split, world_size):
    base_cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    root = base_cfg.embeddings_root
    base_run = f"{split}"
    dst_dir = os.path.join(root, base_run)
    os.makedirs(dst_dir, exist_ok=True)
    dst_h5 = os.path.join(dst_dir, "embeddings.hdf5")
    if os.path.exists(dst_h5):
        os.remove(dst_h5)
    src_dirs = [os.path.join(root, f"{split}-gpu{r}") for r in range(world_size)]
    src_h5s = [os.path.join(d, "embeddings.hdf5") for d in src_dirs if os.path.exists(os.path.join(d, "embeddings.hdf5"))]
    with h5py.File(dst_h5, "w") as tgt:
        for src in src_h5s:
            with h5py.File(src, "r") as sf:
                for gname in sf.keys():
                    if gname in tgt:
                        raise RuntimeError(f"duplicate group {gname}")
                    rel = os.path.relpath(src, dst_dir)
                    tgt[gname] = h5py.ExternalLink(rel, f"/{gname}")
    cfg_path = os.path.join(src_dirs[0], "config.yaml")
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        cfg.run_name = base_run
        OmegaConf.save(cfg, os.path.join(dst_dir, "config.yaml"))
        
def run_split(split, world_size):
    ctx = mp.get_context("spawn")
    ps = []
    for rank in range(world_size):
        p = ctx.Process(target=run_worker, args=(rank, split, world_size))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    merge_split(split, world_size)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for split in ["train", "valid", "test"]:
        run_split(split, world_size)

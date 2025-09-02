import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
import h5py
from omegaconf import OmegaConf
from mammo_datasets import *

def shard_indices(n, world_size, rank):
    q, r = divmod(n, world_size)
    start = rank * q + min(rank, r)
    end = start + q + (1 if rank < r else 0)
    return list(range(start, end))

def run_worker(rank, split, world_size):
    torch.cuda.set_device(rank)
    ds_full = MammoDataset(
        DatasetEnum.EMBED,
        return_mode=ReturnMode.BREAST_LABEL,
        labels=[1, 4, 5, 6],
        convert_to=ConvertTo.RGB_TENSOR,
        split=split,
        final_resize=512,
    )
    idx = shard_indices(len(ds_full), world_size, rank)
    ds = Subset(ds_full, idx)
    cfg = get_embedding_cfg()
    cfg.run_name = f'{split}-gpu{rank}'
    EmbeddingInference(ds, cfg, device=f'cuda:{rank}').run_images()

def merge_split(split, world_size):
    base_cfg = get_embedding_cfg()
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

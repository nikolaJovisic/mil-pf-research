import torch
from dataset import FMDataset, MixMode
import sys
from icecream import ic

pkl_path = "/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl"
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("../head_training")

ds = FMDataset(
    pkl_path=pkl_path,
    mix=MixMode.NONE,
    soft_mix=False,
    groups_per_class=3,
    device=device,
)

for x, y, group, instance_type in ds:
    ic(y, group, instance_type)
    break
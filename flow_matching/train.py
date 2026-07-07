import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os

from dataset import FMDataset, MixMode
# from simple_setflow import SimpleSetFlow
from setflow import SetFlow
from torchinfo import summary
from visualize import monitor
from generate import generate
from utils import compute_fid, compute_interinstance_fid, compute_interlabel_fid, compute_vs_noise_fid, compute_internal_fid, nn_subset_monitor
from icecream import ic


device = "cuda" if torch.cuda.is_available() else "cpu"

pkl_path = "/lustre/nj/cvpr2026/pickles/pca/vindr-v2-128.pkl"

dataset = FMDataset(
    pkl_path=pkl_path,
    mix=MixMode.NONE,
    soft_mix=False,
    groups_per_class=512,
    device=device,
)

model = SetFlow().to(device)
model.train()

optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for step, (x_1, y, group, instance_type) in tqdm(enumerate(dataset)):
    x_0 = torch.randn_like(x_1)

    t = torch.rand(len(x_1), 1, device=device)

    # deterministic
    x_t = (1.0 - t) * x_0 + t * x_1

    # stochastic
    # sigma = 0.1

    # eps = torch.randn_like(x_1)
    # x_t = (1.0 - t) * x_0 + t * x_1 + sigma * torch.sqrt(t * (1 - t)) * eps

    dx_t = x_1 - x_0

    optimizer.zero_grad()

    pred = model(
        x_t=x_t,
        t=t,
        y=y.unsqueeze(-1),
        group=group,
        instance_type=instance_type,
    )

    loss = loss_fn(pred, dx_t)
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        with torch.no_grad():
            v_norm = pred.norm(dim=-1).mean().item()
            x_norm = x_t.norm(dim=-1).mean().item()
            dx_norm = dx_t.norm(dim=-1).mean().item()
            cos = torch.nn.functional.cosine_similarity(
                pred, dx_t, dim=-1
            ).mean().item()


        x_synth, y_synth, group_synth, instance_synth = generate(
            model=model,
            num_bags_y0=2000,
            num_bags_y1=2000,
            feature_dim=128,
            num_steps=50,
            device=device,
        )

        print(
            f"Step {step} | "
            f"loss={loss.item():.4f} | "
            # f"||v||={v_norm:.2f} | "
            # f"||x_t||={x_norm:.2f} | "
            # f"||dx||={dx_norm:.2f} | "
            f"cos(v,dx)={cos:.4f} | "
            f"fid_vs_real={compute_fid(x_1, x_synth):.2f} | "
            f"fid_internal={compute_internal_fid(x_synth)} | "
            # f"fid_vs_noise={compute_vs_noise_fid(x_synth):.2f} | "
            f"interinstance_fid={compute_interinstance_fid(x_synth, instance_synth):.2f} | "
            f"interlabel_fid={compute_interlabel_fid(x_synth, instance_synth, y_synth):.4f} | "
        )

        nn = nn_subset_monitor(x_1, y, group, instance_type, x_synth, y_synth, group_synth, instance_synth)
        ic(nn)

        model.train()
        path = f"weights/vindr-v2-128/setflow_step_{step}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path) 
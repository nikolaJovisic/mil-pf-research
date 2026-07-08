import argparse
import json
import os
from dataclasses import asdict

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from configs import CONFIGS, SetFlowConfig
from dataset import FMDataset, MixMode
from setflow import SetFlow
from generate import generate
from utils import compute_fid, compute_interinstance_fid, compute_interlabel_fid, compute_internal_fid, nn_subset_monitor
from icecream import ic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--pkl_path", default="/lustre/nj/cvpr2026/pickles/pca/vindr-v2-128.pkl")
    parser.add_argument("--out_dir", default="weights/vindr-v2-128")
    return parser.parse_args()


def train(
    config: SetFlowConfig,
    max_steps: int = 100_000,
    eval_every: int = 2000,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
    pkl_path: str = "/lustre/nj/cvpr2026/pickles/pca/vindr-v2-128.pkl",
    out_dir: str = "weights/vindr-v2-128",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = os.path.join(out_dir, config.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    dataset = FMDataset(
        pkl_path=pkl_path,
        mix=MixMode[config.mix_mode],
        soft_mix=config.soft_mix,
        groups_per_class=512,
        device=device,
    )

    model = SetFlow(config).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_internal_fid = float("inf")
    evals_without_improvement = 0

    for step, (x_1, y, group, instance_type) in tqdm(
        enumerate(dataset), total=max_steps, desc=config.name
    ):
        if step >= max_steps:
            break

        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1, device=device)

        if config.stochastic_bridge:
            eps = torch.randn_like(x_1)
            x_t = (1.0 - t) * x_0 + t * x_1 + config.sigma * torch.sqrt(t * (1 - t)) * eps
        else:
            x_t = (1.0 - t) * x_0 + t * x_1

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

        if step % eval_every == 0:
            with torch.no_grad():
                cos = torch.nn.functional.cosine_similarity(
                    pred, dx_t, dim=-1
                ).mean().item()

            x_synth, y_synth, group_synth, instance_synth = generate(
                model=model,
                num_bags_y0=2000,
                num_bags_y1=2000,
                feature_dim=config.dim,
                num_steps=50,
                device=device,
            )

            internal_fid = compute_internal_fid(x_synth)

            print(
                f"[{config.name}] Step {step} | "
                f"loss={loss.item():.4f} | "
                f"cos(v,dx)={cos:.4f} | "
                f"fid_vs_real={compute_fid(x_1, x_synth):.2f} | "
                f"fid_internal={internal_fid:.4f} | "
                f"interinstance_fid={compute_interinstance_fid(x_synth, instance_synth):.2f} | "
                f"interlabel_fid={compute_interlabel_fid(x_synth, instance_synth, y_synth):.4f} | "
            )

            monitor_stats = nn_subset_monitor(
                x_1, y, group, instance_type, x_synth, y_synth, group_synth, instance_synth
            )
            ic(monitor_stats)

            model.train()
            path = os.path.join(save_dir, f"setflow_step_{step}.pth")
            torch.save(model.state_dict(), path)

            if internal_fid < best_internal_fid - early_stop_min_delta:
                best_internal_fid = internal_fid
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= early_stop_patience:
                print(
                    f"[{config.name}] early stopping at step {step}: "
                    f"internal FID did not improve for {early_stop_patience} evaluations "
                    f"(best={best_internal_fid:.4f})"
                )
                break

    del model, optimizer, dataset
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    train(
        config=CONFIGS[args.config],
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

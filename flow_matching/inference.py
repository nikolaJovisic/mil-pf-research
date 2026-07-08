import argparse
import math
import os
import pickle
import sys

import einops
import torch

from configs import CONFIGS
from generate import generate
from setflow import SetFlow

sys.path.append("../head_training")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--weights_dir", default="weights/abl-v2-128")
    parser.add_argument("--input_pkl", default="/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl")
    parser.add_argument("--out_dir", default="synthetic/abl-v2-128")
    return parser.parse_args()


def generate_dataset(model, config, num_bags_y0, num_bags_y1, w_per_class, device):
    x_syn, _, group_syn, instance_type_syn = generate(
        model=model,
        num_bags_y0=num_bags_y0,
        num_bags_y1=num_bags_y1,
        feature_dim=config.dim,
        num_steps=config.num_steps,
        device=device,
    )

    y_syn = torch.cat([torch.zeros(num_bags_y0), torch.ones(num_bags_y1)], dim=0)
    w_syn = torch.cat(
        [
            torch.full((num_bags_y0,), w_per_class[0]),
            torch.full((num_bags_y1,), w_per_class[1]),
        ],
        dim=0,
    )

    return x_syn, y_syn, w_syn, group_syn, instance_type_syn


def run_inference(config_name, weights_dir, input_pkl, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    config = CONFIGS[config_name]
    weights_path = os.path.join(weights_dir, config_name, "setflow.pth")

    save_dir = os.path.join(out_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)
    output_pkl = os.path.join(save_dir, "combined.pkl")
    synthetic_pkl = os.path.join(save_dir, "synthetic.pkl")

    train_ds, valid_ds, test_ds = pickle.load(open(input_pkl, "rb"))
    x_real, y_real, w_real, group_real, instance_type_real = train_ds[0]

    y_real = y_real.view(-1)
    w_real = w_real.view(-1)

    w_per_class = {
        int(c.item()): w_real[y_real == c][0]
        for c in torch.unique(y_real)
    }

    model = SetFlow(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    num_bags_y0 = int((y_real == 0).sum().item())
    num_bags_y1 = int((y_real == 1).sum().item())

    x_syn, y_syn, w_syn, group_syn, instance_type_syn = generate_dataset(
        model=model,
        config=config,
        num_bags_y0=num_bags_y0,
        num_bags_y1=num_bags_y1,
        w_per_class=w_per_class,
        device=device,
    )

    y_syn = einops.rearrange(y_syn, 'b -> b 1')
    w_syn = einops.rearrange(w_syn, 'b -> b 1')

    train_ds_syn = [
        (x_syn, y_syn, w_syn, group_syn, instance_type_syn)
    ]

    y_syn = einops.rearrange(y_syn, 'b 1 -> b')
    w_syn = einops.rearrange(w_syn, 'b 1 -> b')

    pickle.dump(
        (train_ds_syn, valid_ds, test_ds),
        open(synthetic_pkl, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    bonus_bags_y0 = math.ceil(num_bags_y0 * 0.2)
    bonus_bags_y1 = math.ceil(num_bags_y1 * 0.2)

    x_bonus, y_bonus, w_bonus, group_bonus, instance_type_bonus = generate_dataset(
        model=model,
        config=config,
        num_bags_y0=bonus_bags_y0,
        num_bags_y1=bonus_bags_y1,
        w_per_class=w_per_class,
        device=device,
    )

    group_offset = int(group_real.max().item()) + 1
    group_bonus = group_bonus + group_offset

    x_all = torch.cat([x_real, x_bonus])
    group_all = torch.cat([group_real, group_bonus])
    instance_type_all = torch.cat([instance_type_real, instance_type_bonus])

    y_all = einops.rearrange(torch.cat([y_real, y_bonus]), 'b -> b 1')
    w_all = einops.rearrange(torch.cat([w_real, w_bonus]), 'b -> b 1')

    train_ds_aug = [
        (x_all, y_all, w_all, group_all, instance_type_all)
    ]

    pickle.dump(
        (train_ds_aug, valid_ds, test_ds),
        open(output_pkl, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    run_inference(
        config_name=args.config,
        weights_dir=args.weights_dir,
        input_pkl=args.input_pkl,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

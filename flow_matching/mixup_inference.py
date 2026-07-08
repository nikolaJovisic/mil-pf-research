import argparse
import math
import os
import pickle

import einops
import torch

from mixup_configs import CONFIGS, MixupConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="intra_scale_bag", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--input_pkl", default="/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl")
    parser.add_argument("--out_dir", default="synthetic/mixup-v2-128")
    return parser.parse_args()


def _sample_alpha(config: MixupConfig):
    alpha = torch.normal(config.alpha_mean, config.alpha_std, size=(1,))
    return alpha.clamp(config.alpha_min, config.alpha_max).item()


def _build_class_pools(bag_label_per_instance, instance_type, intra_scale):
    pools = {}
    classes = torch.unique(bag_label_per_instance)

    if intra_scale:
        types = torch.unique(instance_type)
        for c in classes:
            for t in types:
                mask = (bag_label_per_instance == c) & (instance_type == t)
                pools[(int(c), int(t))] = mask.nonzero(as_tuple=True)[0]
    else:
        for c in classes:
            mask = (bag_label_per_instance == c)
            pools[int(c)] = mask.nonzero(as_tuple=True)[0]

    return pools


def _sample_partner_from_pool(pool, self_idx, max_tries=3):
    if len(pool) == 0:
        return self_idx
    for _ in range(max_tries):
        partner = pool[torch.randint(0, len(pool), (1,))].item()
        if partner != self_idx:
            return partner
    return self_idx


def _mix_bag(x, bag_instance_idx, instance_type, bag_label, config, class_pools):
    types_template = instance_type[bag_instance_idx]
    n = len(bag_instance_idx)

    new_x = torch.empty(n, x.shape[1], dtype=x.dtype)

    for j in range(n):
        self_idx = int(bag_instance_idx[j].item())
        t = int(types_template[j].item())

        if config.pairing_scope == "bag":
            if config.intra_scale:
                mask = (types_template == t).clone()
            else:
                mask = torch.ones(n, dtype=torch.bool)
            mask[j] = False
            candidates = bag_instance_idx[mask]
            partner_idx = (
                int(candidates[torch.randint(0, len(candidates), (1,))].item())
                if len(candidates) > 0
                else self_idx
            )
        else:  # class
            pool_key = (bag_label, t) if config.intra_scale else bag_label
            partner_idx = _sample_partner_from_pool(class_pools[pool_key], self_idx)

        alpha = _sample_alpha(config)
        new_x[j] = alpha * x[self_idx] + (1 - alpha) * x[partner_idx]

    return new_x


def _bag_instance_index(group, num_bags):
    """Precomputed bag_id -> instance-index lookup, built in one O(N log N)
    pass instead of an O(N) scan per bag (the latter makes bonus-bag
    construction O(n_bonus_bags * N_instances), which is unusably slow on
    large datasets)."""
    order = torch.argsort(group, stable=True)
    counts = torch.bincount(group[order], minlength=num_bags)
    offsets = torch.cumsum(counts, dim=0) - counts
    return order, offsets, counts


def build_bonus_bags(x, y_bag, w_bag, group, instance_type, config, group_offset):
    bag_label_per_instance = y_bag[group]

    class_pools = None
    if config.pairing_scope == "class":
        class_pools = _build_class_pools(bag_label_per_instance, instance_type, config.intra_scale)

    order, offsets, counts = _bag_instance_index(group, num_bags=len(y_bag))

    all_x, all_group, all_instance_type, all_y, all_w = [], [], [], [], []
    next_group_id = group_offset

    for c in torch.unique(y_bag):
        bag_ids_c = (y_bag == c).nonzero(as_tuple=True)[0]
        n_bonus = math.ceil(0.2 * len(bag_ids_c))
        chosen = bag_ids_c[torch.randint(0, len(bag_ids_c), (n_bonus,))]

        for bag_id in chosen:
            bag_id = int(bag_id.item())
            off, cnt = int(offsets[bag_id]), int(counts[bag_id])
            bag_instance_idx = order[off:off + cnt]

            new_x = _mix_bag(x, bag_instance_idx, instance_type, int(c.item()), config, class_pools)
            n = len(bag_instance_idx)

            all_x.append(new_x)
            all_group.append(torch.full((n,), next_group_id, dtype=group.dtype))
            all_instance_type.append(instance_type[bag_instance_idx])
            all_y.append(y_bag[bag_id].unsqueeze(0))
            all_w.append(w_bag[bag_id].unsqueeze(0))

            next_group_id += 1

    return (
        torch.cat(all_x, dim=0),
        torch.cat(all_y, dim=0),
        torch.cat(all_w, dim=0),
        torch.cat(all_group, dim=0),
        torch.cat(all_instance_type, dim=0),
    )


def run_mixup_inference(config_name, input_pkl, out_dir):
    config = CONFIGS[config_name]

    train_ds, valid_ds, test_ds = pickle.load(open(input_pkl, "rb"))
    x_real, y_real, w_real, group_real, instance_type_real = train_ds[0]

    y_real = y_real.view(-1)
    w_real = w_real.view(-1)
    group_real = group_real.view(-1)
    instance_type_real = instance_type_real.view(-1)

    group_offset = int(group_real.max().item()) + 1

    x_bonus, y_bonus, w_bonus, group_bonus, instance_type_bonus = build_bonus_bags(
        x_real, y_real, w_real, group_real, instance_type_real, config, group_offset
    )

    x_all = torch.cat([x_real, x_bonus])
    group_all = torch.cat([group_real, group_bonus])
    instance_type_all = torch.cat([instance_type_real, instance_type_bonus])
    y_all = einops.rearrange(torch.cat([y_real, y_bonus]), 'b -> b 1')
    w_all = einops.rearrange(torch.cat([w_real, w_bonus]), 'b -> b 1')

    train_ds_aug = [
        (x_all, y_all, w_all, group_all, instance_type_all)
    ]

    save_dir = os.path.join(out_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)
    output_pkl = os.path.join(save_dir, "combined.pkl")

    pickle.dump(
        (train_ds_aug, valid_ds, test_ds),
        open(output_pkl, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def main():
    args = parse_args()
    run_mixup_inference(
        config_name=args.config,
        input_pkl=args.input_pkl,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

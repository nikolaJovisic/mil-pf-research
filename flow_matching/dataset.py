import pickle
import torch
from icecream import ic
from enum import Enum
from utils import compute_fm_dataset_stats


class MixMode(Enum):
    NONE = 0
    BAG = 1
    GLOBAL = 2


class FMDataset:
    def __init__(
        self,
        pkl_path,
        groups_per_class=128,
        mix=MixMode.NONE,
        soft_mix=False,
        device=None,
    ):
        self.groups_per_class = groups_per_class
        self.mix = mix
        self.soft_mix = soft_mix

        train_ds, _, _ = pickle.load(open(pkl_path, "rb"))
        self.x, self.y, _, self.group, self.instance_type = train_ds[0]

        self.y = self.y.view(-1)
        self.group = self.group.view(-1)
        self.instance_type = self.instance_type.view(-1)

        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            self.group = self.group.to(device)
            self.instance_type = self.instance_type.to(device)

        compute_fm_dataset_stats(self.y, self.instance_type)

    def sample(self):
        idxs = []
        ys = []
        self.classes = torch.unique(self.y)

        for c in self.classes:
            group_ids = (self.y == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(group_ids), device=self.y.device)
            sampled_groups = group_ids[perm[:self.groups_per_class]]

            mask = torch.isin(self.group, sampled_groups)
            selected_idxs = mask.nonzero(as_tuple=True)[0]

            idxs.append(selected_idxs)
            ys.append(torch.full(
                (selected_idxs.numel(),),
                c,
                device=self.y.device,
                dtype=torch.long,
            ))

        return torch.cat(idxs), torch.cat(ys)



    def mix_batch(self, x, y, group, instance_type):
        if self.mix == MixMode.NONE:
            return x, y

        alpha = torch.rand(len(x), 1, device=x.device)
        perm = torch.arange(len(x), device=x.device)

        for it in torch.unique(instance_type):
            it_idx = (instance_type == it).nonzero(as_tuple=True)[0]

            if self.mix == MixMode.BAG:
                for g in torch.unique(group[it_idx]):
                    m = it_idx[group[it_idx] == g]
                    if len(m) > 1:
                        perm[m] = m[torch.randperm(len(m), device=x.device)]

            elif self.mix == MixMode.GLOBAL:
                if self.soft_mix:
                    perm[it_idx] = it_idx[torch.randperm(len(it_idx), device=x.device)]
                    y = alpha.squeeze(1) * y + (1 - alpha.squeeze(1)) * y[perm]
                else:
                    for c in self.classes:
                        m = it_idx[y[it_idx] == c]
                        if len(m) > 1:
                            perm[m] = m[torch.randperm(len(m), device=x.device)]

        x = alpha * x + (1 - alpha) * x[perm]

        return x, y

    def __iter__(self):
        while True:
            idx, y = self.sample()

            x = self.x[idx]
            it = self.instance_type[idx]

            g = torch.unique_consecutive(self.group[idx], return_inverse=True)[1]

            # this is if we want group-wise y:
            # y = torch.cat([
            #     torch.zeros(self.groups_per_class, device=g.device, dtype=torch.long),
            #     torch.ones(self.groups_per_class, device=g.device, dtype=torch.long),
            # ])

            x, y = self.mix_batch(x, y, g, it)

            yield x, y, g, it

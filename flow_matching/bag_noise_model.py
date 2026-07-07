from enum import Enum
import math
import torch
from torch import nn, Tensor

class BagCond(Enum):
    NONE = 0
    PER_STREAM = 1
    PER_BAG = 2

class TimeEmbedding(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.out_dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device)
        )
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class LowRankFiLM(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden, bias=False)
        self.beta  = nn.Linear(cond_dim, hidden, bias=True)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.gamma(cond) * x + self.beta(cond)

class FlowBranch(nn.Module):
    def __init__(
        self,
        dim,
        hidden,
        cond_dim,
        n_classes,
    ):
        super().__init__()

        self.time_embed  = TimeEmbedding(cond_dim)
        self.class_embed = nn.Embedding(n_classes, cond_dim)

        self.cond_proj = nn.Linear(3 * cond_dim, cond_dim)

        self.fc1   = nn.Linear(dim, hidden)
        self.film1 = LowRankFiLM(hidden, cond_dim)
        self.norm1 = nn.LayerNorm(hidden)

        self.fc2   = nn.Linear(hidden, hidden)
        self.film2 = LowRankFiLM(hidden, cond_dim)
        self.norm2 = nn.LayerNorm(hidden)

        self.fc3   = nn.Linear(hidden, hidden)
        self.film3 = LowRankFiLM(hidden, cond_dim)
        self.norm3 = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x_t, t, y, bag_emb):
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y.squeeze(-1).long())

        cond = self.cond_proj(
            torch.cat([t_emb, y_emb, bag_emb], dim=-1)
        )

        h = self.act(self.norm1(self.film1(self.fc1(x_t), cond)))
        h = self.act(self.norm2(self.film2(self.fc2(h), cond)))
        h = self.act(self.norm3(self.film3(self.fc3(h), cond)))

        return self.out(h)
        
class Flow(nn.Module):
    def __init__(
        self,
        dim=1152,
        hidden=128,
        cond_dim=16,
        n_classes=2,
        bag_cond=BagCond.NONE,
    ):
        super().__init__()

        self.bag_cond = bag_cond
        self.cond_dim = cond_dim

        self.global_stream = FlowBranch(dim, hidden, cond_dim, n_classes)
        self.local_stream = FlowBranch(dim, hidden, cond_dim, n_classes)

    def sample_group_noise(self, group, instance_type):
        B = len(group)
        device = group.device
        z = torch.zeros(B, self.cond_dim, device=device)

        if self.bag_cond == BagCond.PER_BAG:
            for g in torch.unique(group):
                idx = group == g
                z_g = torch.randn(1, self.cond_dim, device=device)
                z[idx] = z_g

        elif self.bag_cond == BagCond.PER_STREAM:
            for g in torch.unique(group):
                for it in torch.unique(instance_type):
                    idx = (group == g) & (instance_type == it)
                    if idx.any():
                        z_gi = torch.randn(1, self.cond_dim, device=device)
                        z[idx] = z_gi

        return z

    def forward(self, x_t, t, y, group, instance_type):
        out = torch.zeros_like(x_t)

        if self.bag_cond == BagCond.NONE:
            bag_emb = torch.zeros(len(x_t), self.cond_dim, device=x_t.device)
        else:
            bag_emb = self.sample_group_noise(group, instance_type)

        mask0 = instance_type == 0
        if mask0.any():
            out[mask0] = self.global_stream(
                x_t[mask0],
                t[mask0],
                y[mask0],
                bag_emb[mask0],
            )

        mask1 = instance_type == 1
        if mask1.any():
            out[mask1] = self.local_stream(
                x_t[mask1],
                t[mask1],
                y[mask1],
                bag_emb[mask1],
            )

        return out

    def step(self, x_t, t_start, t_end, y, group, instance_type):
        dt = t_end - t_start
        t0 = t_start.view(1, 1).expand(len(x_t), 1)

        k1 = self(x_t, t0, y, group, instance_type)
        x_mid = x_t + 0.5 * dt * k1
        k2 = self(x_mid, t0 + 0.5 * dt, y, group, instance_type)

        return x_t + dt * k2

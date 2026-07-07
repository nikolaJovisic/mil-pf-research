import torch
from torch import nn, Tensor
import math


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


class Flow(nn.Module):
    def __init__(
        self,
        dim: int = 1152,
        hidden: int = 128,
        cond_dim: int = 16,
        n_classes: int = 2,
    ):
        super().__init__()

        self.time_embed  = TimeEmbedding(cond_dim)
        self.class_embed = nn.Embedding(n_classes, cond_dim)

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

    def forward(self, x_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c.squeeze(-1).long())
        cond  = t_emb + c_emb

        h = self.fc1(x_t)
        h = self.film1(h, cond)
        h = self.norm1(h)
        h = self.act(h)

        h = self.fc2(h)
        h = self.film2(h, cond)
        h = self.norm2(h)
        h = self.act(h)

        h = self.fc3(h)
        h = self.film3(h, cond)
        h = self.norm3(h)
        h = self.act(h)

        return self.out(h)

    def step(
        self,
        x_t: Tensor,
        t_start: Tensor,
        t_end: Tensor,
        c: Tensor,
    ) -> Tensor:
        dt = t_end - t_start
        t0 = t_start.view(1, 1).expand(x_t.shape[0], 1)

        k1 = self(x_t=x_t, t=t0, c=c)
        x_mid = x_t + 0.5 * dt * k1
        k2 = self(x_t=x_mid, t=t0 + 0.5 * dt, c=c)

        return x_t + dt * k2

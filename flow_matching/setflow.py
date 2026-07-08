import math
import torch
from torch import nn
from torch_scatter import scatter_softmax, scatter_add, scatter_mean
import einops

from configs import SetFlowConfig


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device)
        )
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, hidden, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden, bias=False)
        self.beta = nn.Linear(cond_dim, hidden)

    def forward(self, x, cond):
        return self.gamma(cond) * x + self.beta(cond)


class SegmentISAB(nn.Module):
    def __init__(self, dim, attn_dim, num_inducing):
        super().__init__()

        self.inducing = nn.Parameter(torch.randn(num_inducing, attn_dim))

        self.to_attn = nn.Linear(dim, attn_dim, bias=False)
        self.from_attn = nn.Linear(attn_dim, dim, bias=False)

        self.q_i = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_x = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v_x = nn.Linear(attn_dim, attn_dim, bias=False)

        self.q_x = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_i = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v_i = nn.Linear(attn_dim, attn_dim, bias=False)

    def forward(self, x, group):
        h = self.to_attn(x)

        G = int(group.max()) + 1
        N = x.size(0)
        A = self.q_i.weight.size(0)

        q_i = self.q_i(self.inducing)
        k_x = self.k_x(h)
        v_x = self.v_x(h)

        scores_ix = k_x @ q_i.t() / math.sqrt(A)

        attn_ix = scatter_softmax(scores_ix, group, dim=0)
        z = scatter_add(attn_ix.unsqueeze(-1) * v_x.unsqueeze(1), group, dim=0, dim_size=G)

        z = z[group]

        q_x = self.q_x(h)
        k_i = self.k_i(z)
        v_i = self.v_i(z)

        scores_xi = torch.einsum('nd,nmd->nm', q_x, k_i) / math.sqrt(A)
        attn_xi = torch.softmax(scores_xi, dim=0)
        h2 = torch.einsum('nm,nmd->nd', attn_xi, v_i)

        return self.from_attn(h2)


class SetFlowBlock(nn.Module):
    def __init__(self, config: SetFlowConfig):
        super().__init__()
        self.config = config

        dim = config.dim
        hidden = config.hidden
        cond_dim = config.cond_dim

        self.time_embed = TimeEmbedding(cond_dim)
        self.class_embed = nn.Embedding(config.n_classes, cond_dim)
        self.type_embed = nn.Embedding(config.n_types, cond_dim)

        n_cond_parts = 1 + int(config.use_class_cond) + int(config.use_stream_cond)
        self.cond_proj = nn.Linear(n_cond_parts * cond_dim, cond_dim)

        self.in_proj = nn.Linear(dim, hidden)

        self.film1 = FiLM(hidden, cond_dim)
        self.norm1 = nn.LayerNorm(hidden)

        self.isab = SegmentISAB(hidden, config.attn_dim, config.num_inducing)
        self.norm_isab = nn.LayerNorm(hidden)

        self.pool_proj = nn.Linear(hidden, hidden)

        self.token_mlp = self._build_token_mlp(hidden, config.token_mlp_depth)

        self.film2 = FiLM(hidden, cond_dim)
        self.norm2 = nn.LayerNorm(hidden)

        self.out_proj = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    @staticmethod
    def _build_token_mlp(hidden, depth):
        if depth == 0:
            return nn.Identity()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        return nn.Sequential(*layers)

    def _isab_branch(self, h, group):
        out = self.isab(h, group)
        if self.config.isab_residual:
            return self.norm_isab(h + out)
        return out

    def _pool_branch(self, h, group):
        pooled = scatter_mean(h, group, dim=0)
        return self.pool_proj(pooled[group])

    def _combine(self, h, group):
        mode = self.config.branch_mode
        if mode == "both":
            return self.token_mlp(h) + self._isab_branch(h, group)
        if mode == "mlp_only":
            return self.token_mlp(h)
        if mode == "isab_only":
            return self._isab_branch(h, group)
        if mode == "pool_only":
            return self._pool_branch(h, group)
        if mode == "none":
            return h
        raise ValueError(f"Unknown branch_mode: {mode}")

    def forward(self, x, t, y, instance_type, group):
        t_emb = self.time_embed(t)

        cond_parts = [t_emb]
        if self.config.use_class_cond:
            cond_parts.append(self.class_embed(y.squeeze(-1)))
        if self.config.use_stream_cond:
            cond_parts.append(self.type_embed(instance_type))

        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))

        h = self.in_proj(x)
        h = self.act(self.norm1(self.film1(h, cond)))
        h = self._combine(h, group)

        if self.config.double_film:
            h = self.act(self.norm2(self.film2(h, cond)))
        else:
            h = self.act(self.norm2(h))

        return self.out_proj(h)


class SetFlow(nn.Module):
    def __init__(self, config: SetFlowConfig):
        super().__init__()
        self.config = config
        self.block = SetFlowBlock(config)

    def forward(self, x_t, t, y, group, instance_type):
        return self.block(x_t, t, y, instance_type, group)

    def step(self, x_t, t_start, t_end, y, group, instance_type):
        dt = t_end - t_start
        t0 = t_start.view(1, 1).expand(len(x_t), 1)

        k1 = self(x_t, t0, y, group, instance_type)

        if self.config.integrator == "euler":
            return x_t + dt * k1

        x_mid = x_t + 0.5 * dt * k1
        k2 = self(x_mid, t0 + 0.5 * dt, y, group, instance_type)
        return x_t + dt * k2

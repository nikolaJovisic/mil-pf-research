import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add, scatter_max
from einops import rearrange

class Trex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert cfg.mode in ["global", "local", "both"], "cfg.mode must be one of ['global', 'local', 'both']"

        self.global_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.hidden_dim), nn.ReLU())
        self.global_proj_1 = nn.Sequential(nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim), nn.ReLU())

        self.local_proj_0 = self.global_proj_0 if cfg.share0 else nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.hidden_dim), nn.ReLU())
        self.local_proj_1 = self.global_proj_1 if cfg.share1 else nn.Sequential(nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim), nn.ReLU())

        self.k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        out_dim = 0
        if cfg.mode in ["global", "both"]:
            out_dim += cfg.hidden_dim
        if cfg.mode in ["local", "both"]:
            out_dim += cfg.hidden_dim

        self.linear_out = nn.Linear(out_dim, 1)

    def forward(self, x, group, instance_type):
        outputs = []

        if self.cfg.mode in ["global", "both"]:
            is_whole = instance_type == 0
            x_whole, group_whole = x[is_whole], group[is_whole]
            x_whole = self.global_proj_0(x_whole)
            x_whole = self.global_proj_1(x_whole)
            whole_agg = scatter_max(x_whole, group_whole, dim=0)[0]
            outputs.append(whole_agg)

        if self.cfg.mode in ["local", "both"]:
            is_tile = instance_type == 1
            x_tile, group_tile = x[is_tile], group[is_tile]
            x_tile = self.local_proj_0(x_tile)
            x_tile = self.local_proj_1(x_tile)

            k = self.k(x_tile)
            v = self.v(x_tile)

            G = int(group_tile.max().item()) + 1
            scores = (k @ self.latent.t()).squeeze(-1) / (self.latent.size(-1) ** 0.5)
            attn = segment_softmax(scores, group_tile, num_nodes=G)
            out_group = scatter_add(attn.unsqueeze(-1) * v, group_tile, dim=0, dim_size=G)
            outputs.append(out_group)

        fused = torch.cat(outputs, dim=-1)
        return self.linear_out(fused)

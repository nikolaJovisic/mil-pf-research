import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean

class TrexBothAgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.global_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.gl_hidden_dim), nn.ReLU())
        self.global_proj_1 = nn.Sequential(nn.Linear(2 * cfg.gl_hidden_dim, cfg.gl_hidden_dim), nn.ReLU())

        self.local_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.lc_hidden_dim), nn.ReLU())
        self.local_proj_1 = nn.Sequential(nn.Linear(2 * cfg.lc_hidden_dim, cfg.lc_hidden_dim), nn.ReLU())

        fused_dim = cfg.lc_hidden_dim + cfg.gl_hidden_dim
        self.linear_out = nn.Linear(fused_dim, 1)

    def forward(self, x, group, instance_type):
        is_whole = instance_type == 0
        is_tile = instance_type == 1

        x_whole, group_whole = x[is_whole], group[is_whole]
        x_tile, group_tile = x[is_tile], group[is_tile]

        x_whole = self.global_proj_0(x_whole)
        x_whole = self.global_proj_1(x_whole)
        whole_agg = scatter_max(x_whole, group_whole, dim=0)[0]

        x_tile = self.local_proj_0(x_tile)
        x_tile = self.local_proj_1(x_tile)
        local_agg = scatter_mean(x_tile, group_tile, dim=0) #[0]

        fused = torch.cat([whole_agg, local_agg], dim=-1)
        out = self.linear_out(fused)
        return out

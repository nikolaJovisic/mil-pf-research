import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean

class TrexGlobal(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.global_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.gl_hidden_dim), nn.ReLU())
        self.global_proj_1 = nn.Sequential(nn.Linear(2 * cfg.gl_hidden_dim, cfg.gl_hidden_dim), nn.ReLU())
        self.linear_out = nn.Linear(cfg.gl_hidden_dim, 1)

    def forward(self, x, group, instance_type):
        is_whole = instance_type == 0
        x_whole, group_whole = x[is_whole], group[is_whole]

        x_whole = self.global_proj_0(x)
        x_whole = self.global_proj_1(x_whole)

        whole_agg = scatter_mean(x_whole, group, dim=0)
        out = self.linear_out(whole_agg)
        return out

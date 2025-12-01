import torch
import torch.nn as nn
from torch_scatter import scatter_max

class LinearProbe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear_out = nn.Linear(cfg.input_dim, 1)

    def forward(self, x, group, instance_type):
        pooled = scatter_max(x, group, dim=0)[0]
        return self.linear_out(pooled)


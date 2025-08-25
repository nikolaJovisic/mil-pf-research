import torch
import torch.nn as nn
from torch_scatter import scatter_max

def disassemble_small_state(x):
    pooler = x[:, 0, :]
    cls_ = x[:, 1, :]
    return pooler, cls_

def disassemble_full_state(x):
    pooler = x[:, 0, :]
    cls_ = x[:, 1, :]
    registers = x[:, 2:6, :]
    patch_features_flat = x[:, 6:, :]
    patch_features = patch_features_flat.unflatten(1, (32, 32)) # for 512 inputs
    return pooler, cls_, registers, patch_features

class Dinout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.disassemble_state = disassemble_full_state if cfg.full_state else disassemble_small_state
        self.pick_idx = 0 if cfg.pooler else 1
        if cfg.double_input_layer:
            self.proj = nn.Sequential(
                nn.Linear(cfg.input_dim, 2 * cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.ReLU(),
            )
        self.linear_out = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x, group, _):
        x = self.disassemble_state(x)[self.pick_idx]
        x = self.proj(x)
        x, _ = scatter_max(x, group, dim=0)
        out = self.linear_out(x)
        return out


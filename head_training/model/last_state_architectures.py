import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add, scatter_max
from einops import rearrange

def disassemble_small_state(x):
    pooler = x[:, 0, :]
    cls_ = x[:, 1, :]
    return pooler, cls_

def disassemble_full_state(x):
    pooler = x[:, 0, :]
    cls_ = x[:, 1, :]
    registers = x[:, 2:6, :]
    patch_tokens_flat = x[:, 6:, :]
    patch_tokens = patch_tokens_flat.unflatten(1, (32, 32)) # for 512 inputs
    return pooler, cls_, registers, patch_tokens

class Velo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.global_pick_idx = 0 if cfg.pooler else 1
        self.global_proj = nn.Linear(384, cfg.hidden_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, cfg.hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=7, stride=5),
            nn.ReLU(),
        )
        self.k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))
        self.linear_out = nn.Linear(cfg.hidden_dim * 2, 1)

    def forward(self, x, group):
        state_tuple = disassemble_full_state(x)

        global_token = state_tuple[self.global_pick_idx]
        global_summary = self.global_proj(global_token)
        global_agg, _ = scatter_max(global_summary, group, dim=0)

        patch_tokens = state_tuple[3]
        patch_tokens = rearrange(patch_tokens, 'b h w c -> b c h w')
        patch_tokens = self.conv1(patch_tokens)
        patch_tokens = self.conv2(patch_tokens)
        patch_tokens = rearrange(patch_tokens, 'b c h w -> b (h w) c')

        k = self.k(patch_tokens)
        v = self.v(patch_tokens)

        B, P, C = k.shape
        G = int(group.max().item()) + 1

        g_tok = group[:, None].expand(B, P).reshape(-1)

        scores = (k @ self.latent.t()).squeeze(-1) / (self.latent.size(-1) ** 0.5)
        attn = segment_softmax(scores.reshape(-1), g_tok, num_nodes=G)

        v_flat = v.reshape(B * P, C)
        out_group = scatter_add(attn.unsqueeze(-1) * v_flat, g_tok, dim=0, dim_size=G)

        fused = torch.cat([global_agg, out_group], dim=-1)
        out = self.linear_out(fused)
        return out



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

    def forward(self, x, group):
        x = self.disassemble_state(x)[self.pick_idx]
        x = self.proj(x)
        x, _ = scatter_max(x, group, dim=0)
        out = self.linear_out(x)
        return out
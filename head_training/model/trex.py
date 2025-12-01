import torch
import torch.nn as nn
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add, scatter_max
from einops import rearrange

class Trex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.global_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.gl_hidden_dim), nn.ReLU())
        self.global_proj_1 = nn.Sequential(nn.Linear(2 * cfg.gl_hidden_dim, cfg.gl_hidden_dim), nn.ReLU())

        self.local_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.lc_hidden_dim), nn.ReLU())
        self.local_proj_1 = nn.Sequential(nn.Linear(2 * cfg.lc_hidden_dim, cfg.lc_hidden_dim), nn.ReLU())

        self.k = nn.Linear(cfg.lc_hidden_dim, cfg.lc_hidden_dim)
        self.v = nn.Linear(cfg.lc_hidden_dim, cfg.lc_hidden_dim)

        self.latent = nn.Parameter(torch.randn(cfg.num_latents, cfg.lc_hidden_dim))

        fused_dim = cfg.lc_hidden_dim * cfg.num_latents + cfg.gl_hidden_dim

        if cfg.mlp_out:
            self.linear_out = nn.Sequential(
                nn.Linear(fused_dim, fused_dim// 2),
                nn.ReLU(),
                nn.Linear(fused_dim// 2, 1)
            )
        else:
            self.linear_out = nn.Linear(fused_dim, 1)

    def forward(self, x, group, instance_type, return_attn_maps=False):
        is_whole = instance_type == 0
        is_tile = instance_type == 1

        x_whole, group_whole = x[is_whole], group[is_whole]
        x_tile, group_tile = x[is_tile], group[is_tile]

        x_whole = self.global_proj_0(x_whole)
        x_whole = self.global_proj_1(x_whole)
        whole_agg = scatter_max(x_whole, group_whole, dim=0)[0]

        x_tile = self.local_proj_0(x_tile)
        x_tile = self.local_proj_1(x_tile)

        k = self.k(x_tile)
        v = self.v(x_tile)

        G = int(group_tile.max().item()) + 1

        scores = (k @ self.latent.t()) / (self.latent.size(-1) ** 0.5)
        attn = segment_softmax(scores, group_tile, num_nodes=G)
        if return_attn_maps:
            return attn.reshape(G, self.latent.size(0), -1).cpu().detach()
        out_group = scatter_add(attn.unsqueeze(-1) * v.unsqueeze(1), group_tile, dim=0, dim_size=G)
        out_group = rearrange(out_group, 'g l d -> g (l d)')

        fused = torch.cat([whole_agg, out_group], dim=-1)
        out = self.linear_out(fused)
        return out
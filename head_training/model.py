from enum import Enum
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from torch.nn.functional import softmax


class Aggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
    
def build_model(cfg, device):
    return {
        'global-attends-local': GlobalAttendsLocal,
        'perceiver': Perceiver,
        'baseline': Baseline
    }[cfg.model_type](cfg).to(device)

class Perceiver(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fusion_mode = cfg.fusion_mode
        self.use_shared_projector = cfg.use_shared_projector

        if not self.use_shared_projector:
            cfg.prehidden_dim = cfg.hidden_dim

        self.linear_whole = nn.Linear(cfg.input_dim, cfg.prehidden_dim)
        self.linear_tile = nn.Linear(cfg.input_dim, cfg.prehidden_dim)
        self.relu = nn.ReLU()

        if self.use_shared_projector:
            if cfg.layer_norm:
                self.projector = nn.Sequential(
                    nn.Linear(cfg.prehidden_dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim)
                )
            else:
                self.projector = nn.Sequential(
                    nn.Linear(cfg.prehidden_dim, cfg.hidden_dim)
                )

        self.hidden_dim = cfg.hidden_dim

        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.attention_K = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_V = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.fusion_mode == 'linear':
            self.linear_out = nn.Linear(self.hidden_dim * 2, 1)
        elif self.fusion_mode == 'mlp':
            self.mlp_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.linear_out = nn.Linear(self.hidden_dim, 1)
        elif self.fusion_mode == 'cross-attention':
            self.cross_q = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.cross_k = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.cross_v = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.mlp_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.linear_out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, group, instance_type):
        is_whole = instance_type == 0
        is_tile = instance_type == 1

        x_whole, group_whole = x[is_whole], group[is_whole]
        x_tile, group_tile = x[is_tile], group[is_tile]

        x_whole = self.linear_whole(x_whole)
        x_whole = self.relu(x_whole)

        x_tile = self.linear_tile(x_tile)
        x_tile = self.relu(x_tile)

        if self.use_shared_projector:
            x_whole = self.projector(x_whole)
            x_tile = self.projector(x_tile)

        whole_agg, _ = scatter_max(x_whole, group_whole, dim=0)

        k = self.attention_K(x_tile)
        v = self.attention_V(x_tile)

        group_ids = torch.unique(group_tile)
        outputs = []

        for i, gid in enumerate(group_ids):
            mask = group_tile == gid
            k_g = k[mask]
            v_g = v[mask]

            q_g = self.latent

            scores = (q_g @ k_g.transpose(0, 1)) / q_g.size(-1) ** 0.5
            attn_weights = softmax(scores, dim=-1)
            out_g = attn_weights @ v_g
            out_g = out_g.squeeze(0)

            if self.fusion_mode == 'linear':
                fused = torch.cat([out_g, whole_agg[i]], dim=-1)

            elif self.fusion_mode == 'mlp':
                fused = torch.cat([out_g, whole_agg[i]], dim=-1)
                fused = self.mlp_fusion(fused)

            elif self.fusion_mode == 'cross-attention':
                q = self.cross_q(whole_agg[i].unsqueeze(0))
                k_x = self.cross_k(out_g.unsqueeze(0))
                v_x = self.cross_v(out_g.unsqueeze(0))
                score = (q @ k_x.transpose(0, 1)) / (q.size(-1) ** 0.5)
                attn = softmax(score, dim=-1)
                out = attn @ v_x
                fused = torch.cat([out.squeeze(0), whole_agg[i]], dim=-1)
                fused = self.mlp_fusion(fused)

            outputs.append(fused)

        agg = torch.stack(outputs, dim=0)
        logits = self.linear_out(agg)
        return logits



class GlobalAttendsLocal(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.linear_whole = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.linear_tile = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(cfg.hidden_dim, 1)

        self.attention_Q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attention_K = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attention_V = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            
    def forward(self, x, group, instance_type):
        is_whole = instance_type == 0
        is_tile = instance_type == 1

        x_whole, group_whole = x[is_whole], group[is_whole]
        x_tile, group_tile = x[is_tile], group[is_tile]

        x_whole = self.linear_whole(x_whole)
        x_whole = self.relu(x_whole)

        x_tile = self.linear_tile(x_tile)
        x_tile = self.relu(x_tile)

        q = self.attention_Q(x_whole)
        k = self.attention_K(x_tile)
        v = self.attention_V(x_tile)

        outputs = torch.empty_like(q)
        group_ids = torch.unique(group_whole)

        for gid in group_ids:
            mask_whole = group_whole == gid
            mask_tile = group_tile == gid

            q_g = q[mask_whole]
            k_g = k[mask_tile]
            v_g = v[mask_tile]

            scores = (q_g @ k_g.transpose(0, 1)) / q_g.size(-1)**0.5
            attn_weights = softmax(scores, dim=-1)
            out_g = attn_weights @ v_g

            outputs[mask_whole] = out_g

        agg = scatter_max(outputs, group_whole, dim=0)[0]
        logits = self.linear_out(agg)
        return logits


class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.aggregation = cfg.aggregation

        self.linear_in = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(cfg.hidden_dim, 1)

        if self.aggregation == Aggregation.ATTENTION:
            self.attention_Q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.attention_K = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.attention_V = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            
            
    def forward(self, x, group, instance_type):
        x = self.linear_in(x)
        x = self.relu(x)
        
        agg = self._aggregate(x, group)
        
        logits = self.linear_out(agg)
        return logits


    def _aggregate(self, x, group):
        if self.aggregation == Aggregation.MEAN:
            return scatter_mean(x, group, dim=0)
        elif self.aggregation == Aggregation.MAX:
            max_x, _ = scatter_max(x, group, dim=0)
            return max_x
        elif self.aggregation == Aggregation.ATTENTION:
            q = self.attention_Q(x)
            k = self.attention_K(x)
            v = self.attention_V(x)

            group_ids = torch.unique(group)
            outputs = []

            for gid in group_ids:
                mask = group == gid
                q_g = q[mask]
                k_g = k[mask]
                v_g = v[mask]

                scores = (q_g @ k_g.transpose(0, 1)) / q_g.size(-1)**0.5
                attn_weights = softmax(scores, dim=-1)
                agg = attn_weights @ v_g
                outputs.append(agg.mean(dim=0, keepdim=True))

            return torch.cat(outputs, dim=0)
        else:
            raise ValueError(f"Aggregation type {self.aggregation} not supported.")
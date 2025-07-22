from enum import Enum
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_softmax
from torch.nn.functional import softmax


class Aggregation(Enum):
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"

    
class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cfg.hidden_dim, 1)
        self.aggregation = cfg.aggregation

        self.tile_mode = cfg.use_tiles
        
        self.attention_V = None
        self.attention_w = None

        if self.aggregation == Aggregation.ATTENTION:
            self.attention_V = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.attention_w = nn.Linear(cfg.hidden_dim, 1)

        if self.tile_mode:
            self.tile_attention_V = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.tile_attention_w = nn.Linear(cfg.hidden_dim, 1)
            self.tile_linear = nn.Linear(cfg.hidden_dim, 1)
            self.tile_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, group, instance_type=None):
        x = self.linear1(x)
        x = self.relu(x)

        if self.tile_mode and instance_type is not None:
            is_whole = (instance_type == 0)
            is_tile = (instance_type == 1)

            x_whole, group_whole = x[is_whole], group[is_whole]
            x_tile, group_tile = x[is_tile], group[is_tile]

            agg_whole = self._aggregate(x_whole, group_whole,
                                        self.attention_V, self.attention_w)
            agg_tile = self._aggregate(x_tile, group_tile,
                                       self.tile_attention_V, self.tile_attention_w)

            base_logits = self.linear2(agg_whole)
            tile_logits = self.tile_linear(agg_tile)
            logits = base_logits + self.tile_weight * tile_logits
        else:
            agg = self._aggregate(x, group, self.attention_V, self.attention_w)
            logits = self.linear2(agg)

        return logits

    def _aggregate(self, x, group, V=None, w=None):
        if self.aggregation == Aggregation.MEAN:
            return scatter_mean(x, group, dim=0)
        elif self.aggregation == Aggregation.MAX:
            max_x, _ = scatter_max(x, group, dim=0)
            return max_x
        elif self.aggregation == Aggregation.ATTENTION:
            attn = torch.tanh(V(x))
            scores = w(attn).squeeze(-1)
            weights = scatter_softmax(scores, group, dim=0)
            return scatter_mean(weights.unsqueeze(-1) * x, group, dim=0)
        else:
            raise ValueError(f"Aggregation type {self.aggregation} not supported.")

    def freeze_whole_image_branch(self):
        for param in self.linear1.parameters():
            param.requires_grad = False
        for param in self.linear2.parameters():
            param.requires_grad = False
        if self.aggregation == Aggregation.ATTENTION:
            for param in self.attention_V.parameters():
                param.requires_grad = False
            for param in self.attention_w.parameters():
                param.requires_grad = False


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

        if self.aggregation == Aggregation.ATTENTION:
            self.attention_V = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.attention_w = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x, group):
        x = self.linear1(x)
        x = self.relu(x)
        x = self._aggregate(x, group)
        logits = self.linear2(x)
        return logits

    def _aggregate(self, x, group):
        if self.aggregation == Aggregation.MEAN:
            return scatter_mean(x, group, dim=0)
        elif self.aggregation == Aggregation.MAX:
            max_x, _ = scatter_max(x, group, dim=0)
            return max_x
        elif self.aggregation == Aggregation.ATTENTION:
            attn = torch.tanh(self.attention_V(x))                    # [N, attention_dim]
            scores = self.attention_w(attn).squeeze(-1)               # [N]
            weights = scatter_softmax(scores, group, dim=0)           # [N]
            return scatter_mean(weights.unsqueeze(-1) * x, group, dim=0)  # [num_groups, hidden_dim]
        else:
            raise ValueError(f"Aggregation type {self.aggregation} not supported.")

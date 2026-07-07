import torch
from torch import nn

class SimpleSetFlow(nn.Module):
    def __init__(self, dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t, y=None, group=None, instance_type=None):
        return self.net(torch.cat([t, x_t], dim=-1))

    def step(self, x_t, t_start, t_end, y=None, group=None, instance_type=None):
        dt = t_end - t_start
        t0 = t_start.view(1, 1).expand(len(x_t), 1)

        k1 = self(x_t=x_t, t=t0)
        x_mid = x_t + 0.5 * dt * k1
        k2 = self(x_t=x_mid, t=t0 + 0.5 * dt)

        return x_t + dt * k2

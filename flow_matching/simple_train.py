from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
from icecream import ic

from model import Flow
from utils import (
    sample_alpha,
    fit_pca_lda,
    init_pca_lda_figure,
    add_synth_and_save,
)
from dataset import sample_batch


# def sample_batch(batch_size: int = 64):
#     x_a = torch.randn(batch_size, 1152)
#     x_b = torch.randn(batch_size, 1152)
#     c   = torch.randint(0, 2, (batch_size, 1))
#     return x_a, x_b, c


model = Flow()

total_params = sum(p.numel() for p in model.parameters())
ic(total_params)

optimizer = torch.optim.Adam(model.parameters(), 1e-2)
loss_fn = nn.MSELoss()

with open('msl_gl.pkl', 'rb') as f:
    x, y = pickle.load(f)

ic(x.mean())
ic(x.std())
x = (x - x.mean()) / x.std()

pca, lda = fit_pca_lda(x, y)

fig, ax = init_pca_lda_figure(
    x,
    y,
    pca,
    lda,
    title="initial",
)

eval_res = 100
samples_per_class = 200
time_steps = torch.linspace(0.0, 1.0, 9)

for k, (x_a, x_b, c) in enumerate(tqdm(sample_batch(x, y, batch_size=4096))):
    mask = torch.rand_like(x_a) < 0.5
    x_1 = torch.where(mask, x_a, x_b)

    alpha = sample_alpha(len(x_1), x_1.device)
    x_0 = alpha * x_a + (1 - alpha) * x_b

    t = torch.rand(len(x_1), 1)

    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    
    optimizer.zero_grad()

    pred = model(x_t=x_t, t=t, c=c)
    loss = loss_fn(pred, dx_t)

    loss.backward()
    ic(loss.item())

    optimizer.step()
    
    if k == 100:
        torch.save(model.state_dict(), "flow_matching_model.pth")
        exit(0)

    if k % eval_res == 0:
        xs = []
        ys = []

        for label in [0, 1]:
            idx = (y == label).nonzero(as_tuple=True)[0]
            sel = idx[torch.randperm(len(idx))[:samples_per_class]]
            xs.append(x[sel])
            ys.append(torch.full((samples_per_class,), label))

        x_synth = torch.cat(xs)
        y_synth = torch.cat(ys)

        x_t = x_synth.clone()
        for i in range(len(time_steps) - 1):
            x_t = model.step(
                x_t=x_t,
                t_start=time_steps[i],
                t_end=time_steps[i + 1],
                c=y_synth.unsqueeze(1),
            )

        add_synth_and_save(
            fig,
            ax,
            x_t,
            y_synth,
            pca,
            lda,
            save_path=f"images/k_{k:04d}.png",
        )

        ic(k, loss.item())
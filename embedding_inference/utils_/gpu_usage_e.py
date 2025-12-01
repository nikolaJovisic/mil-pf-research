import time
import torch
from medsiglip_wrapper import build_model

device = "cuda"
model = build_model(device)
batch_size = 1
num_samples = 230000

dummy = [torch.rand(3, 518, 518) for _ in range(batch_size)]

start = time.time()
for i in range(num_samples):
    _ = model(dummy)
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        est_total = avg * num_samples
        hours = est_total / 3600
        print(f"Processed {i+1}/{num_samples}, avg {avg:.4f}s, est total hours {hours:.2f}")







import csv
import torch
import uuid
from datasets_shim import *

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

embed_cfg = {
    get_path('embed'): {'pos': [5, 4], 'neg': [1]},
}

torch.cuda.set_device(0)
run_id = str(uuid.uuid4())[:8]

cfg = load_cfg()

if cfg.flatten and cfg.aggregation == Aggregation.ATTENTION:
    print("Invalid config: flatten=True with ATTENTION aggregation")
    exit()

specificity, sensitivity = train_head(embed_cfg, run_id, cfg)

with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['hidden_dim', 'specificity', 'sensitivity'])
    writer.writerow([cfg.hidden_dim, specificity, sensitivity])
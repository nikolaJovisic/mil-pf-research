import torch
import uuid
import csv
import os
import sys
sys.path.append('..')
from shim import *

def get_dataset_cfg(embedding_id):
    embeddings_root = get_embedding_cfg().embeddings_root

    def get_path(split):
        return f'{embeddings_root}/embed-imagenet-copy/embed-{embedding_id}-{split}/embeddings.hdf5'

    labels = {'pos': [4, 5, 6], 'neg': [1]}

    return {
        'train': {get_path('train'): labels},
        'valid': {get_path('valid'): labels},
        'test': {get_path('train'): labels}
    }

embedding_id = "imagenet"
gpu_id = 0
save_dir = "results_eval"
os.makedirs(save_dir, exist_ok=True)

torch.cuda.set_device(gpu_id)

cfg = load_cfg()
param_combo_id = str(uuid.uuid4())[:8]

report = train_head(
    get_dataset_cfg(embedding_id),
    param_combo_id,
    cfg,
    gpu_id,
    just_evaluate=True
)

summary = report.summary()
summary_keys = report.summary_keys()

with open(f"{save_dir}/eval_{embedding_id}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['embedding_id', 'param_combo_id'] + summary_keys)
    writer.writerow([embedding_id, param_combo_id] + [summary[k] for k in summary_keys])

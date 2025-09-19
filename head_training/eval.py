import torch
import uuid
import csv
import os

from head_training_ import load_cfg, train_head

def get_dataset_cfg(model):
    embeddings_root = '/lustre/nj/dinov3-embeddings/' 

    def get_path(split):
        return f'{embeddings_root}/{model}/{split}/embeddings.hdf5'

    pos_labels = [4, 5, 6]
    neg_labels = [1]

    return {
        'train': ([get_path(f'train-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'valid': ([get_path(f'valid-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'test': ([get_path(f'test-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels)
    }

#save_dir = "results_eval"
#os.makedirs(save_dir, exist_ok=True)

model = 'v2-giant'
cfg = load_cfg()
param_combo_id = str(uuid.uuid4())[:8]

report = train_head(
    get_dataset_cfg(model),
    param_combo_id,
    cfg
    #just_evaluate=True
)

summary = report.summary()
summary_keys = report.summary_keys()

with open(f"{save_dir}/eval_{embedding_id}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['embedding_id', 'param_combo_id'] + summary_keys)
    writer.writerow([embedding_id, param_combo_id] + [summary[k] for k in summary_keys])

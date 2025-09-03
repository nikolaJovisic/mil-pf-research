import csv
import itertools
import multiprocessing as mp
import uuid
import os
import argparse
import torch
import sys

from head_training_ import train_head, load_cfg, Aggregation 
from utils.evaluation_report import EvaluationReport

def get_dataset_cfg(model):
    embeddings_root = '/lustre/nj/dinov3-embeddings/' 

    def get_path(split):
        return f'{embeddings_root}/{model}/{split}/embeddings.hdf5'

    pos_labels = [4, 5, 6]
    neg_labels = [1]

    return {
        'train': (get_path('train'), pos_labels, neg_labels),
        'valid': (get_path('valid'), pos_labels, neg_labels),
        'test': (get_path('test'), pos_labels, neg_labels)
    }

def get_param_grid():
    return {
        'model': ['dinov3-s-512-embed'],
        'hidden_dim': [8, 16, 32, 64, 128],
        'pooler' : [True, False],
    }

def set_nested_attr(obj, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)

def run_training(param_grid, param_list, gpu_id, save_dir):
    import torch
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Assigned combinations:")
    for idx, combo in enumerate(param_list):
        formatted = {
            k: (v.name if hasattr(v, 'name') else v)
            for k, v in combo.items()
        }
        print(f"  [{idx}] {formatted}")

    
    output_file = f"{save_dir}/results_gpu{gpu_id}.csv"
    
    summary_keys = EvaluationReport.summary_keys()

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['param_combo_id'] + list(param_grid.keys()) + summary_keys
        )

    for param_combination in param_list:
        param_combo_id = str(uuid.uuid4())[:8]

        cfg = load_cfg()
        for key_path, value in param_combination.items():
            set_nested_attr(cfg, key_path, value)

        if cfg.flatten and cfg.aggregation == Aggregation.ATTENTION:
            continue

        report = train_head(
            get_dataset_cfg(param_combination['model']),
            param_combo_id,
            cfg,
            gpu_id,
        )

        summary = report.summary()
        row = [param_combo_id] + [
            val.name if hasattr(val, 'name') else val
            for val in param_combination.values()
        ] + [summary[k] for k in summary_keys]

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)



def split_balanced(items, num_splits):
    avg = len(items) / float(num_splits)
    out = []
    last = 0.0

    while last < len(items):
        out.append(items[int(last):int(last + avg)])
        last += avg

    return out

def run_distributed_training(results_dir):
    num_gpus = torch.cuda.device_count()
    
    save_dir = f"{results_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    param_grid = get_param_grid()

    mp.set_start_method('spawn', force=True)

    all_combinations = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    param_dicts = []
    
    for combo in all_combinations:
        combo_dict = dict(zip(keys, combo))
        param_dicts.append(combo_dict)

    chunks = split_balanced(param_dicts, num_gpus)

    processes = []
    for gpu_id, param_list in enumerate(chunks):
        p = mp.Process(target=run_training, args=(param_grid, param_list, gpu_id, save_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default="results")
    args = parser.parse_args()

    run_distributed_training(args.results_dir)


if __name__ == "__main__":
    main()

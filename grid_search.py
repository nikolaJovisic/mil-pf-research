import csv
import itertools
import multiprocessing as mp
from datasets_shim import *

def get_path(dataset):
    return f'/home/nikola.jovisic.ivi/nj/lustre_mock/{dataset}/embeddings.hdf5'

embed_cfg = {
    get_path('embed'): {'pos': [5, 4], 'neg': [1]},
}

param_grid = {
    'flatten': [True, False],
    'aggregation': [Aggregation.ATTENTION, Aggregation.MAX, Aggregation.MEAN],
    'hidden_dim': [8, 16, 32],
}

def set_nested_attr(obj, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)

def run_training(param_list, gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Assigned combinations:")
    for idx, combo in enumerate(param_list):
        formatted = {
            k: (v.name if hasattr(v, 'name') else v)
            for k, v in combo.items()
        }
        print(f"  [{idx}] {formatted}")

    output_file = f"results_gpu{gpu_id}.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(param_grid.keys()) + ['specificity', 'sensitivity'])

    for param_combination in param_list:
        cfg = load_cfg()
        for key_path, value in param_combination.items():
            set_nested_attr(cfg, key_path, value)
            
        if cfg.flatten and cfg.aggregation == Aggregation.ATTENTION:
            continue

        specificity, sensitivity = train_head(
            embed_cfg,
            cfg,
            gpu_id
        )

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                val.name if hasattr(val, 'name') else val
                for val in param_combination.values()
            ] + [specificity, sensitivity]
            writer.writerow(row)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    all_combinations = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    param_dicts = [dict(zip(keys, combo)) for combo in all_combinations]

    num_gpus = 6
    chunks = [[] for _ in range(num_gpus)]
    for i, combo in enumerate(param_dicts):
        chunks[i % num_gpus].append(combo)

    processes = []
    for gpu_id, param_list in enumerate(chunks):
        p = mp.Process(target=run_training, args=(param_list, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

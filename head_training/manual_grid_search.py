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
# from utils.multiclass_evaluation_report import EvaluationReport

def get_param_grid():
    return {
       'runs': list(range(36)),
    }

def set_nested_attr(obj, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)

def run_training(param_grid, param_list, gpu_id, save_dir, pickle_path=None):
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
            ['param_combo_id'] + list(param_grid.keys()) + summary_keys + ['valid_auc'] + ['train_auc'] + ['valid_spec_90']
        )

    cfg = load_cfg()
    print('Config loaded.')

    if pickle_path is not None:
        cfg.pickle_path = pickle_path

    for param_combination in param_list:
        param_combo_id = str(uuid.uuid4())[:8]
        #cfg.save_path = f"/lustre/nj/cvpr2026/head_weights/{param_combo_id}.pth"

        for key_path, value in param_combination.items():
            set_nested_attr(cfg, key_path, value)

        if cfg.flatten and cfg.aggregation == Aggregation.ATTENTION:
            continue

        test_report, valid_report, train_report = train_head(
            param_combo_id,
            cfg,
            gpu_id,
        )

        test_summary = test_report.summary()
        valid_summary = valid_report.summary()
        train_summary = train_report.summary()
        
        row = [param_combo_id] + [
            val.name if hasattr(val, 'name') else val
            for val in param_combination.values()
        ] + [test_summary[k] for k in summary_keys] + [valid_summary['auc']] + [train_summary['auc']] + [valid_summary['spec_90']]
        # ] + [test_summary[k] for k in summary_keys] + [valid_summary['accuracy']] + [train_summary['accuracy']]

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

def run_distributed_training(results_dir, pickle_path=None):
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
        # if combo_dict['lc_hidden_dim'] > combo_dict['gl_hidden_dim']:
        #     continue
        param_dicts.append(combo_dict)

    chunks = split_balanced(param_dicts, num_gpus)


    processes = []
    for gpu_id, param_list in enumerate(chunks):
        p = mp.Process(target=run_training, args=(param_grid, param_list, gpu_id, save_dir, pickle_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default="results")
    parser.add_argument('--pickle-path', type=str, default=None)
    args = parser.parse_args()

    run_distributed_training(args.results_dir, args.pickle_path)


if __name__ == "__main__":
    main()

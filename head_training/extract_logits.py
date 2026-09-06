import argparse
import multiprocessing as mp
import os
import pickle

import torch

from head_training_ import _train, load_cfg
from utils.evaluation_report import EvaluationReport


def collect_logits(model, dataset, device):
    # Mirrors head_training.utils.evaluate.evaluate()'s loop exactly, but
    # stops short of the sigmoid + EvaluationReport wrapping: raw logits
    # aren't exposed anywhere else in the training/eval pipeline.
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, _, group, instance_type in dataset:
            x, group, instance_type = x.to(device), group.to(device), instance_type.to(device)
            logits = model(x, group, instance_type)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)
    return logits, labels


def train_once(cfg, datasets, device, run_id):
    train_ds, valid_ds, test_ds = datasets

    log_dir = os.path.join(cfg.logs_path, run_id)
    os.makedirs(log_dir, exist_ok=True)

    model = _train(train_ds, valid_ds, cfg, device, os.path.join(log_dir, "train.csv"),
                   just_evaluate=False, run_id=run_id)

    logits, labels = collect_logits(model, test_ds, device)

    # same specificity-at-90%-sensitivity operating point as Table 5, so the
    # plotting side can mark where this run's own decision threshold falls
    # in logit space.
    report = EvaluationReport(torch.sigmoid(logits), labels)
    spec_90 = report.specificity_at(0.9)
    prob_threshold = next(
        (t for t in torch.linspace(1, 0, EvaluationReport.LINSPACE_STEPS).tolist() if report.sensitivity(t) >= 0.9),
        0.5,
    )
    return {
        "logits": logits,
        "labels": labels,
        # plain floats, not the numpy scalars sklearn hands back, so the
        # file stays loadable under torch.load's weights_only default
        "auc": float(report.auc()),
        "spec_90": float(spec_90),
        "prob_threshold_90": float(prob_threshold),
    }


def worker(pickle_path, run_indices, gpu_id, run_id_prefix, runs_dir, reuse):
    # cfg is rebuilt here rather than passed in, so nothing OmegaConf-shaped
    # has to survive the spawn pickle
    cfg = load_cfg()
    cfg.pickle_path = pickle_path
    # a fresh classifier trained only on this pickle's train split, exactly
    # like each manual_grid_search.py run -- not fine-tuned from anything,
    # and not persisted anywhere else.
    cfg.load_path = None
    cfg.save_path = None

    device = "cpu"
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

    datasets = pickle.load(open(cfg.pickle_path, "rb"))

    for idx in run_indices:
        out_path = os.path.join(runs_dir, f"run{idx}.pt")
        if reuse and os.path.exists(out_path):
            print(f"[gpu {gpu_id}] run {idx}: reusing {out_path}", flush=True)
            continue

        result = train_once(cfg, datasets, device, f"{run_id_prefix}_r{idx}")
        result["run"] = idx
        # written as each run lands, so a crash partway through a sweep keeps
        # whatever already finished
        torch.save(result, out_path)
        print(
            f"[gpu {gpu_id}] run {idx}: AUC={result['auc']:.3f}, Spec@90={result['spec_90']:.3f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default="logit_extraction")
    # 8 to match get_param_grid()'s {'runs': range(8)} in manual_grid_search.py:
    # every number in the paper is the best of 8 such trainings, so anything
    # less would put these figures on a different footing than the tables.
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    parser.add_argument("--reuse-runs", action="store_true", help="skip runs already on disk")
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    runs_dir = os.path.splitext(out_path)[0] + "_runs"
    os.makedirs(runs_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    num_workers = max(1, min(num_gpus, args.runs))
    run_indices = list(range(args.runs))

    if num_workers <= 1:
        worker(args.pickle_path, run_indices, 0 if num_gpus >= 1 else None,
               args.run_id, runs_dir, args.reuse_runs)
    else:
        mp.set_start_method("spawn", force=True)
        # strided rather than manual_grid_search.py's split_balanced(), which
        # can hand back more chunks than there are GPUs; this always yields
        # exactly num_workers of them
        chunks = [run_indices[i::num_workers] for i in range(num_workers)]
        procs = [
            mp.Process(target=worker, args=(args.pickle_path, chunk, gpu_id,
                                            args.run_id, runs_dir, args.reuse_runs))
            for gpu_id, chunk in enumerate(chunks)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            raise RuntimeError(f"{len(bad)} of {len(procs)} workers exited non-zero: {bad}")

    results = []
    for idx in run_indices:
        path = os.path.join(runs_dir, f"run{idx}.pt")
        if os.path.exists(path):
            results.append(torch.load(path, weights_only=False))
    if not results:
        raise RuntimeError(f"no runs completed under {runs_dir}")

    aucs = torch.tensor([r["auc"] for r in results])
    specs = torch.tensor([r["spec_90"] for r in results])
    # best-by-test-AUC, the same selection fill_ablations.best_test_metrics()
    # makes over manual_grid_search.py's results_gpu*.csv
    best = max(results, key=lambda r: r["auc"])

    # the spread is what says whether a gap between two regimes is real or
    # just seed noise, so carry it alongside the selected run
    best["runs"] = len(results)
    best["auc_all"] = [r["auc"] for r in results]
    best["spec_90_all"] = [r["spec_90"] for r in results]

    torch.save(best, out_path)
    print(
        f"Saved {best['logits'].numel()} test-set logits from the best of {len(results)} runs "
        f"to {out_path}\n"
        f"  best : AUC={best['auc']:.3f}, Spec@90={best['spec_90']:.3f} (run {best['run']})\n"
        f"  AUC   over runs: mean={aucs.mean():.3f} std={aucs.std(unbiased=False):.3f} "
        f"min={aucs.min():.3f} max={aucs.max():.3f}\n"
        f"  Spec@90 over runs: mean={specs.mean():.3f} std={specs.std(unbiased=False):.3f} "
        f"min={specs.min():.3f} max={specs.max():.3f}"
    )


if __name__ == "__main__":
    main()

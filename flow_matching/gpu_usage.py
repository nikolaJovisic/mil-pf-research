import argparse
import json
import os
import time

import torch

from configs import CONFIGS, SetFlowConfig
from inference import run_inference
from mixup_configs import CONFIGS as MIXUP_CONFIGS
from mixup_inference import run_mixup_inference
from train import train

DEFAULT_PICKLES = {
    "msl": "/lustre/nj/cvpr2026/pickles/pca/msl-128.pkl",
    "v2": "/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl",
}

ENCODER_LABELS = {"msl": "MedSigLIP", "v2": "DINOv2"}

# configs.py's CONFIGS registry only lists ablation variants (the plain
# baseline is commented out of _configs_list); register it here so
# inference.run_inference("baseline", ...) can look it up like any other
# named config, without touching configs.py itself.
BASELINE_CONFIG = SetFlowConfig(name="baseline")
CONFIGS.setdefault("baseline", BASELINE_CONFIG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", nargs="*", default=["train", "inference", "mixup"],
        choices=["train", "inference", "mixup"],
    )
    parser.add_argument("--encoders", nargs="*", default=["msl", "v2"], choices=sorted(DEFAULT_PICKLES))
    parser.add_argument("--pkl_msl", default=DEFAULT_PICKLES["msl"])
    parser.add_argument("--pkl_v2", default=DEFAULT_PICKLES["v2"])
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--weights_dir", default="weights/gpu_usage")
    parser.add_argument("--synthetic_dir", default="synthetic/gpu_usage")
    parser.add_argument("--mixup_config", default="intra_scale_bag", choices=sorted(MIXUP_CONFIGS))
    parser.add_argument("--mixup_pkl", default=DEFAULT_PICKLES["msl"])
    parser.add_argument("--mixup_dir", default="synthetic/gpu_usage-mixup")
    parser.add_argument("--results_jsonl", default="results/gpu_usage/gpu_usage.jsonl")
    parser.add_argument("--tex_path", default="tables/gpu_usage.tex")
    return parser.parse_args()


def pkl_for(encoder, args):
    return args.pkl_msl if encoder == "msl" else args.pkl_v2


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reset_gpu_stats(device):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb(device):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _timed(device, fn, **kwargs):
    _reset_gpu_stats(device)
    start = time.perf_counter()
    fn(**kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, _peak_vram_gb(device)


def _record(stage, encoder, device, elapsed, peak_vram_gb):
    record = {
        "stage": stage,
        "encoder": encoder,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "seconds": elapsed,
        "gpu_hours": elapsed / 3600 if device == "cuda" else None,
        "peak_vram_gb": peak_vram_gb,
    }
    vram_msg = f", peak VRAM {peak_vram_gb:.2f} GB" if peak_vram_gb is not None else ""
    print(f"[gpu_usage] {stage}/{encoder}: {elapsed / 3600:.3f} h wall-clock{vram_msg}")
    return record


def measure_train(encoder, args, device):
    pkl_path = pkl_for(encoder, args)
    out_dir = os.path.join(args.weights_dir, encoder)
    print(f"[gpu_usage] training baseline SetFlow on {ENCODER_LABELS[encoder]} ({pkl_path})")
    elapsed, peak_vram = _timed(
        device, train,
        config=BASELINE_CONFIG,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        pkl_path=pkl_path,
        out_dir=out_dir,
    )
    return _record("train", encoder, device, elapsed, peak_vram)


def measure_inference(encoder, args, device):
    pkl_path = pkl_for(encoder, args)
    weights_path = os.path.join(args.weights_dir, encoder, "baseline", "setflow.pth")
    if not os.path.exists(weights_path):
        print(
            f"[gpu_usage] skipping inference for {encoder}: no checkpoint at {weights_path} "
            "(run the 'train' stage first)"
        )
        return None

    print(f"[gpu_usage] running inference (synthetic + 20% bonus bags) for {ENCODER_LABELS[encoder]}")
    elapsed, peak_vram = _timed(
        device, run_inference,
        config_name="baseline",
        weights_dir=os.path.join(args.weights_dir, encoder),
        input_pkl=pkl_path,
        out_dir=os.path.join(args.synthetic_dir, encoder),
    )
    return _record("inference", encoder, device, elapsed, peak_vram)


def measure_mixup(args):
    print(f"[gpu_usage] running MixUp ({args.mixup_config}) baseline on CPU")
    elapsed, _ = _timed(
        "cpu", run_mixup_inference,
        config_name=args.mixup_config,
        input_pkl=args.mixup_pkl,
        out_dir=args.mixup_dir,
    )
    return _record("mixup", "both", "cpu", elapsed, None)


def append_results(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            if r is not None:
                f.write(json.dumps(r) + "\n")


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_by(records, key_fields):
    # a re-run of a stage should overwrite its stale entry rather than
    # duplicate a row in the table, so keep only the last record per key
    latest = {}
    for r in records:
        latest[tuple(r[k] for k in key_fields)] = r
    return list(latest.values())


def fmt(v, spec=".2f"):
    return "--" if v is None else format(v, spec)


def write_tex(records, tex_path):
    records = latest_by(records, ["stage", "encoder"])
    by_key = {(r["stage"], r["encoder"]): r for r in records}

    gpu_name = next((r["gpu_name"] for r in records if r.get("gpu_name")), None)
    gpu_note = f"an NVIDIA {gpu_name}" if gpu_name else "a single GPU"

    lines = [
        "\\begin{table}[H]",
        "\\caption{Computational cost of SetFlow training and inference against a MixUp "
        "baseline, measured on " + gpu_note + ". Wall-clock and GPU-hours coincide since a "
        "single device is used; peak VRAM is the maximum value returned by "
        "\\texttt{torch.cuda.max\\_memory\\_allocated} during the run.\\label{tab:gpu_usage}}",
        "\\begin{tabularx}{\\textwidth}{llCCC}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Stage} & \\textbf{Wall-clock (h)} & \\textbf{GPU-hours} "
        "& \\textbf{Peak VRAM (GB)} \\\\",
        "\\midrule",
    ]

    for encoder in ["msl", "v2"]:
        label = ENCODER_LABELS[encoder]
        train_r = by_key.get(("train", encoder))
        infer_r = by_key.get(("inference", encoder))

        lines.append(f"\\multirow{{2}}{{*}}{{{label}}}")
        lines.append(
            "& Training (SetFlow) & "
            f"{fmt(train_r['seconds'] / 3600) if train_r else '--'} & "
            f"{fmt(train_r['gpu_hours']) if train_r else '--'} & "
            f"{fmt(train_r['peak_vram_gb']) if train_r else '--'} \\\\"
        )
        lines.append(
            "& Inference (SetFlow) & "
            f"{fmt(infer_r['seconds'] / 3600) if infer_r else '--'} & "
            f"{fmt(infer_r['gpu_hours']) if infer_r else '--'} & "
            f"{fmt(infer_r['peak_vram_gb']) if infer_r else '--'} \\\\"
        )
        lines.append("\\midrule")

    mixup_r = by_key.get(("mixup", "both"))
    lines.append(
        "MixUp (CPU only) & Inference & "
        f"{fmt(mixup_r['seconds'] / 3600) if mixup_r else '--'} & -- & -- \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")

    tex_dir = os.path.dirname(tex_path)
    if tex_dir:
        os.makedirs(tex_dir, exist_ok=True)
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = device_name()

    new_records = []

    if "train" in args.stages:
        for encoder in args.encoders:
            new_records.append(measure_train(encoder, args, device))

    if "inference" in args.stages:
        for encoder in args.encoders:
            new_records.append(measure_inference(encoder, args, device))

    if "mixup" in args.stages:
        new_records.append(measure_mixup(args))

    append_results(new_records, args.results_jsonl)

    all_records = load_results(args.results_jsonl)
    write_tex(all_records, args.tex_path)
    print(f"[gpu_usage] wrote {args.tex_path}")


if __name__ == "__main__":
    main()

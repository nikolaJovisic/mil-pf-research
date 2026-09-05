import argparse
import os
import subprocess
import sys

import torch

HEAD_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "head_training"))

REGIME_LABELS = {"original": "Original", "synthetic": "Synthetic-only"}
REGIME_COLORS = {"original": "tab:blue", "synthetic": "tab:orange"}
CLASS_LABELS = {0: "Benign (y=0)", 1: "Malignant (y=1)"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="*", default=["msl", "v2"], choices=["msl", "v2"])
    # No defaults: which pickle is "the" original/synthetic-only pickle for
    # a given encoder depends on which inference.py run produced it, and
    # guessing wrong here would silently compare the wrong distributions.
    parser.add_argument("--original_pkl_msl")
    parser.add_argument("--synthetic_pkl_msl")
    parser.add_argument("--original_pkl_v2")
    parser.add_argument("--synthetic_pkl_v2")
    parser.add_argument("--work_dir", default="results/logit_analysis")
    parser.add_argument("--out_dir", default="logit_figs")
    parser.add_argument("--skip_extraction", action="store_true", help="reuse .pt files already in --work_dir")
    parser.add_argument("--extract_only", action="store_true", help="save logits without plotting them")
    return parser.parse_args()


def pickle_paths(encoder, args):
    return {
        "original": getattr(args, f"original_pkl_{encoder}"),
        "synthetic": getattr(args, f"synthetic_pkl_{encoder}"),
    }


def extract(encoder, regime, pkl_path, work_dir):
    out_path = os.path.join(work_dir, f"{encoder}_{regime}.pt")
    print(f"=== {encoder}/{regime}: training classifier on {pkl_path} ===")
    subprocess.run(
        [
            sys.executable, "extract_logits.py",
            "--pickle-path", os.path.abspath(pkl_path),
            "--out", os.path.abspath(out_path),
            "--run-id", f"logit_{encoder}_{regime}",
        ],
        cwd=HEAD_TRAINING_DIR,
        check=True,
    )
    return out_path


def plot_class_density(encoder, cls, runs, out_dir):
    # imported here, not at module scope, so a broken plotting stack (e.g. a
    # matplotlib built against NumPy 1.x on a NumPy 2 machine) can't block the
    # GPU-side extraction, whose results are already saved by then.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))

    for regime, data in runs.items():
        logits = data["logits"][data["labels"] == cls]
        if len(logits) < 2:
            continue

        plt.hist(
            logits.numpy(), bins=30, density=True, alpha=0.5,
            color=REGIME_COLORS[regime], label=REGIME_LABELS[regime],
        )

        logit_threshold = torch.logit(torch.tensor(data["prob_threshold_90"])).item()
        plt.axvline(logit_threshold, color=REGIME_COLORS[regime], linestyle="--", linewidth=1)

    plt.xlabel("Classifier logit")
    plt.ylabel("Density")
    plt.title(f"{encoder.upper()} test-set logits -- {CLASS_LABELS[cls]}\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(out_dir, f"{encoder}_class{cls}_logits.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"wrote {filename}")


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for encoder in args.encoders:
        pkls = pickle_paths(encoder, args)
        missing = [regime for regime, path in pkls.items() if not path]
        if missing:
            print(f"=== skipping {encoder}: missing --{'/'.join(f'{r}_pkl_{encoder}' for r in missing)} ===")
            continue

        runs = {}
        for regime, pkl_path in pkls.items():
            out_path = os.path.join(args.work_dir, f"{encoder}_{regime}.pt")
            if not (args.skip_extraction and os.path.exists(out_path)):
                out_path = extract(encoder, regime, pkl_path, args.work_dir)
            runs[regime] = torch.load(out_path)
            print(
                f"  {encoder}/{regime}: AUC={runs[regime]['auc']:.3f}, "
                f"Spec@90={runs[regime]['spec_90']:.3f}"
            )

        if args.extract_only:
            continue

        for cls in (0, 1):
            plot_class_density(encoder, cls, runs, args.out_dir)


if __name__ == "__main__":
    main()

import argparse
import math
import os
import subprocess
import sys

import torch

HEAD_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "head_training"))

# order matters: it fixes the legend/plot order in every figure
REGIME_LABELS = {"original": "Original", "combined": "Combined", "synthetic": "Synthetic-only"}
REGIME_COLORS = {"original": "tab:blue", "combined": "tab:green", "synthetic": "tab:orange"}
CLASS_LABELS = {0: "Benign (y=0)", 1: "Malignant (y=1)"}

GRID_POINTS = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="*", default=["msl", "v2"], choices=["msl", "v2"])
    # No defaults: which pickle is "the" original/combined/synthetic-only pickle
    # for a given encoder depends on which inference.py run produced it, and
    # guessing wrong here would silently compare the wrong distributions.
    parser.add_argument("--original_pkl_msl")
    parser.add_argument("--combined_pkl_msl")
    parser.add_argument("--synthetic_pkl_msl")
    parser.add_argument("--original_pkl_v2")
    parser.add_argument("--combined_pkl_v2")
    parser.add_argument("--synthetic_pkl_v2")
    parser.add_argument("--work_dir", default="results/logit_analysis")
    parser.add_argument("--out_dir", default="logit_figs")
    parser.add_argument("--skip_extraction", action="store_true", help="reuse .pt files already in --work_dir")
    parser.add_argument("--extract_only", action="store_true", help="save logits without plotting them")
    parser.add_argument("--runs", type=int, default=8, help="trainings per regime; best test AUC wins, as in manual_grid_search.py")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs to spread those trainings over (default: all visible)")
    parser.add_argument("--reuse_runs", action="store_true", help="keep runs already on disk instead of retraining them")
    parser.add_argument("--per_class", action="store_true", help="also write the per-class logit histograms")
    return parser.parse_args()


def pickle_paths(encoder, args):
    return {regime: getattr(args, f"{regime}_pkl_{encoder}") for regime in REGIME_LABELS}


def extract(encoder, regime, pkl_path, work_dir, args):
    out_path = os.path.join(work_dir, f"{encoder}_{regime}.pt")
    print(f"=== {encoder}/{regime}: training classifier on {pkl_path} ===")
    cmd = [
        sys.executable, "extract_logits.py",
        "--pickle-path", os.path.abspath(pkl_path),
        "--out", os.path.abspath(out_path),
        "--run-id", f"logit_{encoder}_{regime}",
        "--runs", str(args.runs),
    ]
    if args.gpus is not None:
        cmd += ["--gpus", str(args.gpus)]
    if args.reuse_runs:
        cmd += ["--reuse-runs"]
    subprocess.run(cmd, cwd=HEAD_TRAINING_DIR, check=True)
    return out_path


# --- curves, in plain torch -------------------------------------------------
# sklearn would give all three of these, but it is the one import that kept
# breaking on the cluster's NumPy build, and it is only ever called here on a
# few thousand scores, so the closed forms are cheaper than the dependency.

def _ranked(scores, labels):
    # sort by score descending and keep one point per *distinct* score, so tied
    # logits collapse into a single threshold instead of drawing a staircase
    order = torch.argsort(scores, descending=True)
    s, y = scores[order], labels[order].float()
    tp = torch.cumsum(y, 0)
    fp = torch.cumsum(1.0 - y, 0)
    keep = torch.cat([s[1:] != s[:-1], torch.tensor([True])])
    return tp[keep], fp[keep], y.sum(), (1.0 - y).sum()


def roc_curve(scores, labels):
    tp, fp, pos, neg = _ranked(scores, labels)
    zero = torch.zeros(1)
    fpr = torch.cat([zero, fp / neg])
    tpr = torch.cat([zero, tp / pos])
    return fpr, tpr, torch.trapz(tpr, fpr).item()


def pr_curve(scores, labels):
    tp, fp, pos, _ = _ranked(scores, labels)
    recall = tp / pos
    precision = tp / (tp + fp)
    # average precision as the step-wise sum, i.e. sklearn's average_precision_score
    # rather than the trapezoid, which is optimistic on a jagged PR curve
    ap = (torch.cat([recall[:1], recall[1:] - recall[:-1]]) * precision).sum().item()
    return recall, precision, ap


def kde(samples, grid):
    # Gaussian KDE with Silverman's rule; the pooled logits are bimodal, so a
    # smooth curve reads far better than three overlapping histograms
    bandwidth = 1.06 * samples.std().item() * samples.numel() ** -0.2
    z = (grid[:, None] - samples[None, :]) / bandwidth
    return torch.exp(-0.5 * z ** 2).sum(1) / (samples.numel() * bandwidth * math.sqrt(2 * math.pi))


def _pyplot():
    # imported lazily, not at module scope, so a broken plotting stack (e.g. a
    # matplotlib built against NumPy 1.x on a NumPy 2 machine) can't block the
    # GPU-side extraction, whose results are already saved by then.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, out_dir, filename):
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"wrote {path}")


def plot_logit_density(encoder, runs, out_dir):
    plt = _pyplot()
    plt.figure(figsize=(6, 4))

    lo = min(data["logits"].min().item() for data in runs.values())
    hi = max(data["logits"].max().item() for data in runs.values())
    pad = 0.05 * (hi - lo)
    grid = torch.linspace(lo - pad, hi + pad, GRID_POINTS)

    for regime, data in runs.items():
        # both classes pooled: this is the full test-set score distribution the
        # Spec@90 threshold is actually swept over
        density = kde(data["logits"], grid)
        color = REGIME_COLORS[regime]
        plt.plot(grid.numpy(), density.numpy(), color=color, lw=1.8, label=REGIME_LABELS[regime])
        plt.fill_between(grid.numpy(), density.numpy(), color=color, alpha=0.12)

        logit_threshold = torch.logit(torch.tensor(data["prob_threshold_90"])).item()
        plt.axvline(logit_threshold, color=color, linestyle="--", linewidth=1)

    plt.xlabel("Classifier logit")
    plt.ylabel("Density")
    plt.title(f"{encoder.upper()} test-set logits, both classes pooled\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
    plt.legend()
    _save(plt, out_dir, f"{encoder}_logits.png")


def plot_roc(encoder, runs, out_dir):
    plt = _pyplot()
    plt.figure(figsize=(5, 5))

    for regime, data in runs.items():
        fpr, tpr, auc = roc_curve(data["logits"], data["labels"])
        color = REGIME_COLORS[regime]
        plt.plot(fpr.numpy(), tpr.numpy(), color=color, lw=1.8, label=f"{REGIME_LABELS[regime]} (AUC={auc:.3f})")

        # the Sens=0.9 operating point Table 5 reports specificity at
        idx = int((tpr >= 0.9).nonzero()[0])
        plt.plot(fpr[idx].item(), tpr[idx].item(), "o", color=color, ms=5)

    plt.axhline(0.9, color="0.6", linestyle=":", linewidth=1)
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(f"{encoder.upper()} ROC\n(markers = Sens=0.9 operating point)")
    plt.legend(loc="lower right")
    _save(plt, out_dir, f"{encoder}_roc.png")


def plot_pr(encoder, runs, out_dir):
    plt = _pyplot()
    plt.figure(figsize=(5, 5))

    for regime, data in runs.items():
        recall, precision, ap = pr_curve(data["logits"], data["labels"])
        color = REGIME_COLORS[regime]
        plt.plot(recall.numpy(), precision.numpy(), color=color, lw=1.8, label=f"{REGIME_LABELS[regime]} (AP={ap:.3f})")

        idx = int((recall >= 0.9).nonzero()[0])
        plt.plot(recall[idx].item(), precision[idx].item(), "o", color=color, ms=5)

    # all three regimes are evaluated on the same real test split, so one
    # prevalence line covers every curve
    labels = next(iter(runs.values()))["labels"]
    prevalence = labels.float().mean().item()
    plt.axhline(prevalence, color="0.6", linestyle="--", linewidth=1, label=f"Chance (prev.={prevalence:.3f})")

    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"{encoder.upper()} precision-recall\n(markers = Recall=0.9 operating point)")
    plt.legend(loc="lower left")
    _save(plt, out_dir, f"{encoder}_pr.png")


def plot_class_density(encoder, cls, runs, out_dir):
    plt = _pyplot()
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
    _save(plt, out_dir, f"{encoder}_class{cls}_logits.png")


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for encoder in args.encoders:
        pkls = pickle_paths(encoder, args)
        missing = [regime for regime, path in pkls.items() if not path]
        if missing:
            print(f"=== skipping {encoder}: missing {', '.join('--%s_pkl_%s' % (r, encoder) for r in missing)} ===")
            continue

        runs = {}
        for regime, pkl_path in pkls.items():
            out_path = os.path.join(args.work_dir, f"{encoder}_{regime}.pt")
            if not (args.skip_extraction and os.path.exists(out_path)):
                out_path = extract(encoder, regime, pkl_path, args.work_dir, args)
            # weights_only=False so files written before the float() fix in
            # extract_logits.py, which carry numpy scalars, still load
            runs[regime] = torch.load(out_path, weights_only=False)
            data = runs[regime]
            line = f"  {encoder}/{regime}: AUC={data['auc']:.3f}, Spec@90={data['spec_90']:.3f}"
            if "auc_all" in data:
                # the reported run is the best of N; the spread over all N is
                # what says whether a gap between two regimes is real noise
                aucs = torch.tensor(data["auc_all"])
                specs = torch.tensor(data["spec_90_all"])
                line += (
                    f"  (best of {data['runs']}; "
                    f"AUC {aucs.mean():.3f}+-{aucs.std(unbiased=False):.3f}, "
                    f"Spec@90 {specs.mean():.3f}+-{specs.std(unbiased=False):.3f})"
                )
            print(line)

        if args.extract_only:
            continue

        plot_logit_density(encoder, runs, args.out_dir)
        plot_roc(encoder, runs, args.out_dir)
        plot_pr(encoder, runs, args.out_dir)

        if args.per_class:
            for cls in (0, 1):
                plot_class_density(encoder, cls, runs, args.out_dir)


if __name__ == "__main__":
    main()

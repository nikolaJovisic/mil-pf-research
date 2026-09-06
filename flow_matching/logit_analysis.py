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
CLASS_COLORS = {0: "tab:blue", 1: "tab:red"}

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
    # --skip_extraction is the original spelling, kept working so existing
    # invocations don't break
    parser.add_argument("--reuse_logits", "--reuse-logits", "--skip_extraction", dest="reuse_logits",
                        action="store_true",
                        help="plot straight from the .pt logits already in --work_dir, training nothing")
    parser.add_argument("--extract_only", action="store_true", help="save logits without plotting them")
    parser.add_argument("--trim", type=float, default=0.05,
                        help="fraction clipped off each tail when choosing the x range (0 to disable)")
    parser.add_argument("--runs", type=int, default=8, help="trainings per regime; best test AUC wins, as in manual_grid_search.py")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs to spread those trainings over (default: all visible)")
    parser.add_argument("--reuse_runs", "--reuse-runs", dest="reuse_runs", action="store_true", help="keep runs already on disk instead of retraining them")
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
    # smooth curve reads far better than three overlapping histograms.
    # Silverman's robust form, min(std, IQR/1.349), rather than std alone: a
    # handful of saturated logits inflates std enough to smooth the bell flat.
    s = samples.float()
    q1, q3 = torch.quantile(s, torch.tensor([0.25, 0.75])).tolist()
    spread = min(s.std().item(), (q3 - q1) / 1.349) or s.std().item()
    bandwidth = 0.9 * spread * s.numel() ** -0.2
    if bandwidth <= 0:
        bandwidth = 1.0
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


def _bounds(samples, trim):
    if trim <= 0:
        return samples.min().item(), samples.max().item()
    q = torch.tensor([trim, 1.0 - trim])
    lo, hi = torch.quantile(samples.float(), q).tolist()
    return lo, hi


def _grid(sample_sets, trim, include=()):
    """x range covering each set's central 1-2*trim mass, plus `include` points.

    Taken per set and then unioned rather than over the pooled samples, so a
    regime with a wide spread cannot swallow a narrow one -- and clipped by
    quantile rather than min/max, because a few saturated logits at +-30 would
    otherwise squeeze every bell into a spike at the origin.
    """
    bounds = [_bounds(t, trim) for t in sample_sets if t.numel() >= 2]
    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    for v in include:
        # the decision threshold has to stay on screen even if it sits out in
        # a trimmed tail, which is exactly the case worth seeing
        lo, hi = min(lo, v), max(hi, v)
    pad = 0.05 * (hi - lo)
    return torch.linspace(lo - pad, hi + pad, GRID_POINTS)


def _curve(ax, samples, grid, color, label):
    if samples.numel() < 2:
        return
    density = kde(samples, grid)
    ax.plot(grid.numpy(), density.numpy(), color=color, lw=1.8, label=label)
    ax.fill_between(grid.numpy(), density.numpy(), color=color, alpha=0.12)


def _threshold(data):
    return torch.logit(torch.tensor(data["prob_threshold_90"])).item()


def _class_panel(ax, data, grid, title):
    # each class normalised on its own, so the separation stays readable even
    # though benign outnumbers malignant roughly 3:1 in the test split
    for cls, color in CLASS_COLORS.items():
        _curve(ax, data["logits"][data["labels"] == cls], grid, color, CLASS_LABELS[cls])
    ax.axvline(_threshold(data), color="0.35", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Classifier logit")


def plot_pooled(encoder, runs, out_dir, grid):
    """One curve per regime, both classes pooled."""
    plt = _pyplot()
    plt.figure(figsize=(6, 4))

    for regime, data in runs.items():
        color = REGIME_COLORS[regime]
        _curve(plt.gca(), data["logits"], grid, color, REGIME_LABELS[regime])
        plt.axvline(_threshold(data), color=color, linestyle="--", linewidth=1)

    plt.xlabel("Classifier logit")
    plt.ylabel("Density")
    plt.title(f"{encoder.upper()} test-set logits, both classes pooled\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
    plt.legend()
    _save(plt, out_dir, f"{encoder}_logits_pooled.png")


def plot_regime_classes(encoder, regime, data, out_dir, trim):
    """One regime, benign vs malignant, scaled to its own spread."""
    plt = _pyplot()
    # standalone figure, so it gets its own range: nothing here is being
    # compared across regimes, and a shared range only wastes the axis
    grid = _grid([data["logits"][data["labels"] == c] for c in CLASS_LABELS],
                 trim, include=[_threshold(data)])
    fig, ax = plt.subplots(figsize=(6, 4))
    _class_panel(ax, data, grid, f"{encoder.upper()} {REGIME_LABELS[regime]}")
    ax.set_ylabel("Density (per class)")
    ax.legend()
    _save(plt, out_dir, f"{encoder}_logits_{regime}.png")


def plot_regime_grid(encoder, runs, out_dir, grid):
    """All three regimes side by side, each split by class, on shared axes."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharex=True, sharey=True)
    axes = axes if hasattr(axes, "__len__") else [axes]

    for ax, (regime, data) in zip(axes, runs.items()):
        _class_panel(ax, data, grid, REGIME_LABELS[regime])

    axes[0].set_ylabel("Density (per class)")
    axes[0].legend()
    fig.suptitle(f"{encoder.upper()} test-set logits by class (dashed = that regime's Spec@Sens=0.9 threshold)")
    _save(plt, out_dir, f"{encoder}_logits_grid.png")


def plot_class_across_regimes(encoder, cls, runs, out_dir, grid):
    """One class, all three regimes overlaid."""
    plt = _pyplot()
    plt.figure(figsize=(6, 4))

    for regime, data in runs.items():
        color = REGIME_COLORS[regime]
        _curve(plt.gca(), data["logits"][data["labels"] == cls], grid, color, REGIME_LABELS[regime])
        plt.axvline(_threshold(data), color=color, linestyle="--", linewidth=1)

    plt.xlabel("Classifier logit")
    plt.ylabel("Density")
    plt.title(f"{encoder.upper()} test-set logits -- {CLASS_LABELS[cls]}\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
    plt.legend()
    _save(plt, out_dir, f"{encoder}_logits_class{cls}.png")


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


def plot_all(encoder, runs, out_dir, trim):
    # one grid shared by every figure that puts regimes side by side, so those
    # stay directly comparable; the standalone per-regime figures scale
    # themselves instead
    grid = _grid(
        [d["logits"][d["labels"] == c] for d in runs.values() for c in CLASS_LABELS],
        trim,
        include=[_threshold(d) for d in runs.values()],
    )

    plot_pooled(encoder, runs, out_dir, grid)
    plot_regime_grid(encoder, runs, out_dir, grid)
    for regime, data in runs.items():
        plot_regime_classes(encoder, regime, data, out_dir, trim)
    for cls in CLASS_LABELS:
        plot_class_across_regimes(encoder, cls, runs, out_dir, grid)
    plot_roc(encoder, runs, out_dir)
    plot_pr(encoder, runs, out_dir)


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
            if not (args.reuse_logits and os.path.exists(out_path)):
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

        plot_all(encoder, runs, args.out_dir, args.trim)


if __name__ == "__main__":
    main()

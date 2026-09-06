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
# short names used to build the per-regime --reuse-*-logits flags
REGIME_SHORT = {"original": "orig", "combined": "comb", "synthetic": "synth"}

# the Sens=0.9 operating point, drawn in one fixed colour rather than each
# curve's own: it has to read as an annotation on top of the curves, not as
# another series
OP_MARKER = dict(marker="X", markersize=11, color="black",
                 markeredgecolor="white", markeredgewidth=1.4,
                 linestyle="none", zorder=5)
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
    # per-regime equivalents, so a single regime can be recomputed while the
    # other two are read back off disk -- generated from REGIME_SHORT rather
    # than written out, so a new regime cannot pick up a flag by accident
    for _regime, _short in REGIME_SHORT.items():
        parser.add_argument(f"--reuse_{_short}_logits", f"--reuse-{_short}-logits",
                            dest=f"reuse_{_regime}", action="store_true",
                            help=f"reuse the saved {_regime} logits instead of retraining them")
        parser.add_argument(f"--drop_{_short}", f"--drop-{_short}",
                            dest=f"drop_{_regime}", action="store_true",
                            help=f"leave {_regime} out of the run entirely -- not trained, not plotted")
    parser.add_argument("--extract_only", action="store_true", help="save logits without plotting them")
    parser.add_argument("--roc_zoom_x", "--roc-zoom-x", dest="roc_zoom_x", nargs=2, type=float,
                        default=[0.8, 1.0], metavar=("LO", "HI"),
                        help="x (false positive rate) limits of the zoomed ROC")
    parser.add_argument("--roc_zoom_y", "--roc-zoom-y", dest="roc_zoom_y", nargs=2, type=float,
                        default=[0.9, 1.0], metavar=("LO", "HI"),
                        help="y (sensitivity) limits of the zoomed ROC")
    parser.add_argument("--trim", type=float, default=0.05,
                        help="fraction clipped off each tail when choosing the x range (0 to disable)")
    parser.add_argument("--runs", type=int, default=8, help="trainings per regime; best test AUC wins, as in manual_grid_search.py")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs to spread those trainings over (default: all visible)")
    parser.add_argument("--reuse_runs", "--reuse-runs", dest="reuse_runs", action="store_true", help="keep runs already on disk instead of retraining them")
    return parser.parse_args()


def active_regimes(args):
    active = [r for r in REGIME_LABELS if not getattr(args, f"drop_{r}")]
    if not active:
        raise ValueError("every regime dropped; nothing left to plot")
    return active


def reused_regimes(args):
    if args.reuse_logits:
        return set(REGIME_LABELS)
    return {r for r in REGIME_LABELS if getattr(args, f"reuse_{r}")}


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
    plt.title("Test-set logits, both classes pooled\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
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
    _class_panel(ax, data, grid, REGIME_LABELS[regime])
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
    fig.suptitle("Test-set logits by class (dashed = that regime's Spec@Sens=0.9 threshold)")
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
    plt.title(f"Test-set logits -- {CLASS_LABELS[cls]}\n(dashed = each regime's own Spec@Sens=0.9 threshold)")
    plt.legend()
    _save(plt, out_dir, f"{encoder}_logits_class{cls}.png")


def plot_roc(encoder, runs, out_dir, xlim=None, ylim=None):
    zoom = xlim is not None or ylim is not None
    plt = _pyplot()
    plt.figure(figsize=(5, 5))

    for regime, data in runs.items():
        fpr, tpr, _ = roc_curve(data["logits"], data["labels"])
        plt.plot(fpr.numpy(), tpr.numpy(), color=REGIME_COLORS[regime], lw=1.8,
                 label=REGIME_LABELS[regime])

        # the Sens=0.9 operating point Table 5 reports specificity at
        idx = int((tpr >= 0.9).nonzero()[0])
        plt.plot(fpr[idx].item(), tpr[idx].item(), **OP_MARKER)

    plt.axhline(0.9, color="0.6", linestyle=":", linewidth=1)
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")

    if zoom:
        # drawn from the full curves and then clipped, so the visible segment is
        # the real one -- not a curve recomputed on a subset of thresholds
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        # the whole point of the zoom is reading small vertical gaps
        plt.grid(True, alpha=0.3, linewidth=0.6)

    plt.title(("ROC, zoomed" if zoom else "ROC") + "\n(X = Sens=0.9 operating point)")
    plt.legend(loc="lower right")
    _save(plt, out_dir, f"{encoder}_roc{'_zoom' if zoom else ''}.png")


def plot_pr(encoder, runs, out_dir):
    plt = _pyplot()
    plt.figure(figsize=(5, 5))

    for regime, data in runs.items():
        recall, precision, _ = pr_curve(data["logits"], data["labels"])
        plt.plot(recall.numpy(), precision.numpy(), color=REGIME_COLORS[regime], lw=1.8,
                 label=REGIME_LABELS[regime])

        idx = int((recall >= 0.9).nonzero()[0])
        plt.plot(recall[idx].item(), precision[idx].item(), **OP_MARKER)

    # all three regimes are evaluated on the same real test split, so one
    # prevalence line covers every curve
    labels = next(iter(runs.values()))["labels"]
    prevalence = labels.float().mean().item()
    plt.axhline(prevalence, color="0.6", linestyle="--", linewidth=1, label="Chance")

    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-recall\n(X = Recall=0.9 operating point)")
    plt.legend(loc="lower left")
    _save(plt, out_dir, f"{encoder}_pr.png")


def plot_all(encoder, runs, out_dir, trim, roc_zoom_x=None, roc_zoom_y=None):
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
    plot_roc(encoder, runs, out_dir, xlim=roc_zoom_x, ylim=roc_zoom_y)
    plot_pr(encoder, runs, out_dir)


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    active = active_regimes(args)
    reuse = reused_regimes(args)
    if len(active) < len(REGIME_LABELS):
        print(f"=== plotting {', '.join(active)} only ===")

    for encoder in args.encoders:
        pkls = pickle_paths(encoder, args)

        # work out which regimes actually need training before demanding any
        # pickle paths, so reusing two of them means only the third one's
        # --*_pkl_* has to be passed at all
        saved = {}
        needed = []
        for regime in active:
            out_path = os.path.join(args.work_dir, f"{encoder}_{regime}.pt")
            if regime in reuse:
                if os.path.exists(out_path):
                    saved[regime] = out_path
                    continue
                print(f"  [warn] {encoder}/{regime}: asked to reuse but {out_path} is missing; retraining it")
            needed.append(regime)

        missing = [r for r in needed if not pkls[r]]
        if missing:
            print(f"=== skipping {encoder}: missing {', '.join('--%s_pkl_%s' % (r, encoder) for r in missing)} ===")
            continue
        if saved:
            print(f"=== {encoder}: reusing {', '.join(sorted(saved))}; training {', '.join(needed) or 'nothing'} ===")

        runs = {}
        for regime in active:
            out_path = saved.get(regime)
            if out_path is None:
                out_path = extract(encoder, regime, pkls[regime], args.work_dir, args)
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

        plot_all(encoder, runs, args.out_dir, args.trim,
                 roc_zoom_x=args.roc_zoom_x, roc_zoom_y=args.roc_zoom_y)


if __name__ == "__main__":
    main()

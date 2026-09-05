import argparse
import os

from fill_ablations import run_config, update_tex

# maps the pickle subdirectory written by augmentation_ratio.py -> row label
# as it appears in augmentation_ratio.tex
ROW_LABELS = {
    "ratio_05": "5\\%",
    "ratio_10": "10\\%",
    "ratio_20": "20\\%",
    "ratio_50": "50\\%",
    "ratio_100": "100\\%",
}

PICKLE_DIRS = {
    "msl": "/lustre/nj/cvpr2026/pickles/setflow/ratios-msl-128",
    "v2": "/lustre/nj/cvpr2026/pickles/setflow/ratios-v2-128",
}


def ratio_name(ratio):
    # must match augmentation_ratio.py, which names the pickle subdirectories;
    # kept local so this script needs no torch/einops just to build a path
    return f"ratio_{round(ratio * 100):02d}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="fill the DINOv2 columns")
    parser.add_argument("--msl", action="store_true", help="fill the MedSigLIP columns")
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.05, 0.1])
    parser.add_argument("--msl_pickles_dir", default=PICKLE_DIRS["msl"])
    parser.add_argument("--v2_pickles_dir", default=PICKLE_DIRS["v2"])
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--results_dir", default="results/ratios")
    parser.add_argument("--tex_path", default="tables/augmentation_ratio.tex")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.v2 and not args.msl:
        raise ValueError("Pass --v2 and/or --msl.")

    names = [ratio_name(r) for r in args.ratios]
    unknown = [n for n in names if n not in ROW_LABELS]
    if unknown:
        raise ValueError(f"No table row for: {', '.join(unknown)}")

    embeddings = [
        ("msl", "MedSigLIP", args.msl_pickles_dir),
        ("v2", "DINOv2", args.v2_pickles_dir),
    ]

    for embedding, label, pickles_dir in embeddings:
        if not getattr(args, embedding):
            continue

        print(f"=== {label} augmentation ratios ===")
        for name in names:
            # augmentation_ratio.py writes to <pickles_dir>/<ratio>/<config>/,
            # since run_inference always appends the config name
            metrics = run_config(
                args.config,
                os.path.join(pickles_dir, name),
                os.path.join(args.results_dir, embedding, name),
            )
            if metrics is None:
                continue

            update_tex(args.tex_path, {embedding: {name: metrics}}, ROW_LABELS)
            print(f"  updated {args.tex_path} with '{name}'")


if __name__ == "__main__":
    main()

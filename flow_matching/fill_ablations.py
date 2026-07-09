import argparse
import glob
import os
import re
import subprocess
import sys

import pandas as pd

HEAD_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "head_training"))

# maps SetFlow ablation config name (configs.py) -> row label as it appears in ablation.tex
ROW_LABELS = {
    "isab_only": "ISAB only",
    "mlp_only": "TokenMLP only",
    "pool_only": "Pool only",
    "token_mlp_depth_1": "TokenMLP depth 1",
    "token_mlp_depth_5": "TokenMLP depth 5",
    "no_stream_cond": "No stream cond.",
    "single_film": "Single FiLM",
    "num_inducing_1": "\\# inducing 1",
    "num_inducing_8": "\\# inducing 8",
    "cond_dim_8": "cond. dim 8",
    "cond_dim_32": "cond. dim 32",
}

BACC_COL = "bal_acc_w0.5"
SPEC_COL = "spec_90"

ROW_PATTERN = re.compile(
    r'^(?P<label>.+?)\s*&\s*(?P<c1>[^&]*)&\s*(?P<c2>[^&]*)&\s*(?P<c3>[^&]*)&'
    r'\s*(?P<c4>[^&]*)&\s*(?P<c5>[^&]*)&\s*(?P<c6>[^&]*)\\\\\s*$'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="fill the DINOv2 columns")
    parser.add_argument("--msl", action="store_true", help="fill the MedSigLIP columns")
    parser.add_argument(
        "--v2_pickles_dir",
        default="/lustre/data/cvrs.mammo.ivi/nj/cvpr2026/pickles/setflow/abl-v2-128",
    )
    parser.add_argument(
        "--msl_pickles_dir",
        default="/lustre/data/cvrs.mammo.ivi/nj/cvpr2026/pickles/setflow/abl-msl-128",
    )
    parser.add_argument("--results_dir", default="results/ablations")
    parser.add_argument("--tex_path", default="ablation.tex")
    return parser.parse_args()


def best_test_metrics(results_dir):
    csv_paths = sorted(glob.glob(os.path.join(results_dir, "results_gpu*.csv")))
    if not csv_paths:
        return None

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if df.empty:
        return None

    best = df.loc[df["auc"].idxmax()]
    return float(best["auc"]), float(best[BACC_COL]), float(best[SPEC_COL])


def run_config(config_name, pickle_root, results_root):
    combined_pkl = os.path.abspath(os.path.join(pickle_root, config_name, "combined.pkl"))
    if not os.path.exists(combined_pkl):
        print(f"  [skip] {combined_pkl} not found")
        return None

    config_results_dir = os.path.abspath(os.path.join(results_root, config_name))
    os.makedirs(config_results_dir, exist_ok=True)

    print(f"  running grid search for '{config_name}' on {combined_pkl}")
    # run as a subprocess (not in-process) so manual_grid_search.py's bare
    # `from utils...` imports resolve against its own directory instead of
    # colliding with flow_matching/utils.py, which would otherwise shadow
    # head_training/utils/ once both dirs are on the same sys.path.
    subprocess.run(
        [
            sys.executable, "manual_grid_search.py",
            "--results-dir", config_results_dir,
            "--pickle-path", combined_pkl,
        ],
        cwd=HEAD_TRAINING_DIR,
        check=True,
    )

    metrics = best_test_metrics(config_results_dir)
    if metrics is None:
        print(f"  [warn] no results produced for '{config_name}'")
    return metrics


def format_val(v):
    return f"{v:.3f}"


def update_tex(tex_path, updates_by_embedding):
    label_to_config = {v: k for k, v in ROW_LABELS.items()}

    with open(tex_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        m = ROW_PATTERN.match(line.rstrip("\n"))
        label = m.group("label").strip() if m else None

        if m and label in label_to_config:
            config_name = label_to_config[label]
            cells = [m.group(f"c{i}").strip() for i in range(1, 7)]

            for embedding, results in updates_by_embedding.items():
                metrics = results.get(config_name)
                if metrics is None:
                    continue
                auc, bacc, spec = metrics
                vals = [format_val(auc), format_val(bacc), format_val(spec)]
                if embedding == "msl":
                    cells[0:3] = vals
                else:
                    cells[3:6] = vals

            new_lines.append(
                f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & "
                f"{cells[3]} & {cells[4]} & {cells[5]} \\\\\n"
            )
        else:
            new_lines.append(line)

    with open(tex_path, "w") as f:
        f.writelines(new_lines)


def main():
    args = parse_args()
    if not args.v2 and not args.msl:
        raise ValueError("Pass --v2 and/or --msl.")

    updates_by_embedding = {}

    if args.msl:
        print("=== MedSigLIP ablations ===")
        results = {}
        for config_name in ROW_LABELS:
            results[config_name] = run_config(
                config_name, args.msl_pickles_dir, os.path.join(args.results_dir, "msl")
            )
        updates_by_embedding["msl"] = results

    if args.v2:
        print("=== DINOv2 ablations ===")
        results = {}
        for config_name in ROW_LABELS:
            results[config_name] = run_config(
                config_name, args.v2_pickles_dir, os.path.join(args.results_dir, "v2")
            )
        updates_by_embedding["v2"] = results

    update_tex(args.tex_path, updates_by_embedding)
    print(f"Updated {args.tex_path}")


if __name__ == "__main__":
    main()

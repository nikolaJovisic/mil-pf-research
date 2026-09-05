import argparse
import os

from configs import CONFIGS
from inference import run_inference

# Ranked by damage to bACC in the existing Synthetic-Only ablation
# (Table \ref{tab:ablation}), averaged across MedSigLIP/DINOv2, most
# destructive first. This answers the reviewer's "repeat the key ablation
# variants (at least the top 3 most destructive ones) in the Combined
# regime" request in Section 5.6: the first 3 entries are that minimum,
# the other 3 let the authors extend coverage without adding a config.
DEFAULT_CONFIGS = [
    "token_mlp_depth_5",  # avg bACC drop 0.177 (0.698->0.500 MedSigLIP, 0.699->0.544 DINOv2)
    "no_stream_cond",     # avg bACC drop 0.169 (both encoders collapse toward chance)
    "cond_dim_8",         # avg bACC drop 0.129 (DINOv2 bACC collapses to 0.500)
    "token_mlp_depth_1",  # avg bACC drop 0.128 (DINOv2 bACC/Spec. collapse to 0.500/0.000)
    "isab_only",          # avg bACC drop 0.122 (worst of the branch-mode variants)
    "single_film",        # avg bACC drop 0.108; the reviewer's own example -- MedSigLIP
                           # AUC stays intact (0.732) while bACC/Spec. collapse (0.506/0.003),
                           # i.e. a shifted operating point rather than lost ranking ability,
                           # which is exactly the kind of effect real-data mixing could mask
                           # or fix.
]

# Same weights_dir/input_pkl convention as run_ablations.py / run_ablations_inference.py.
ENCODERS = {
    "msl": dict(
        weights_dir="weights/abl-msl-128",
        input_pkl="/lustre/nj/cvpr2026/pickles/pca/msl-128.pkl",
        out_dir="/lustre/nj/cvpr2026/pickles/setflow/abl-msl-128-combined-review",
    ),
    "v2": dict(
        weights_dir="weights/abl-v2-128",
        input_pkl="/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl",
        out_dir="/lustre/nj/cvpr2026/pickles/setflow/abl-v2-128-combined-review",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--encoders", nargs="*", default=sorted(ENCODERS), choices=sorted(ENCODERS))
    parser.add_argument("--weights_dir_msl", default=ENCODERS["msl"]["weights_dir"])
    parser.add_argument("--weights_dir_v2", default=ENCODERS["v2"]["weights_dir"])
    parser.add_argument("--pkl_msl", default=ENCODERS["msl"]["input_pkl"])
    parser.add_argument("--pkl_v2", default=ENCODERS["v2"]["input_pkl"])
    # Deliberately a *separate* tree from the original abl-{msl,v2}-128 pickle
    # dirs that Table 7 was built from: run_inference() also regenerates
    # synthetic.pkl (stochastic) alongside combined.pkl here, so
    # pointing this at the original out_dir would silently redraw and
    # overwrite the samples the paper's synthetic-only numbers came from.
    parser.add_argument("--out_dir_msl", default=ENCODERS["msl"]["out_dir"])
    parser.add_argument("--out_dir_v2", default=ENCODERS["v2"]["out_dir"])
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")

    paths = {
        "msl": dict(weights_dir=args.weights_dir_msl, input_pkl=args.pkl_msl, out_dir=args.out_dir_msl),
        "v2": dict(weights_dir=args.weights_dir_v2, input_pkl=args.pkl_v2, out_dir=args.out_dir_v2),
    }

    for encoder in args.encoders:
        p = paths[encoder]
        for name in args.configs:
            weights_path = os.path.join(p["weights_dir"], name, "setflow.pth")
            if not os.path.exists(weights_path):
                print(f"=== skipping {encoder}/{name}: no weights found at {weights_path} ===")
                continue

            print(f"=== {encoder}: generating Combined-regime pickle for '{name}' ===")
            run_inference(
                config_name=name,
                weights_dir=p["weights_dir"],
                input_pkl=p["input_pkl"],
                out_dir=p["out_dir"],
            )


if __name__ == "__main__":
    main()

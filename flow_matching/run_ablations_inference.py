import argparse
import os

from configs import CONFIGS
from inference import run_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=sorted(CONFIGS.keys()))
    parser.add_argument("--weights_dir", default="weights/abl-v2-128")
    parser.add_argument("--input_pkl", default="/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl")
    parser.add_argument("--out_dir", default="/lustre/nj/cvpr2026/pickles/setflow/abl-v2-128")
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")

        weights_path = os.path.join(args.weights_dir, name, "setflow.pth")
        if not os.path.exists(weights_path):
            print(f"=== skipping config '{name}': no weights found at {weights_path} ===")
            continue

        print(f"=== running inference for config '{name}' ===")
        run_inference(
            config_name=name,
            weights_dir=args.weights_dir,
            input_pkl=args.input_pkl,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

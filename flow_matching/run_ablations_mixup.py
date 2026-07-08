import argparse

from mixup_configs import CONFIGS
from mixup_inference import run_mixup_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=sorted(CONFIGS.keys()))
    parser.add_argument("--input_pkl", default="/lustre/nj/cvpr2026/pickles/pca/v2-128.pkl")
    parser.add_argument("--out_dir", default="synthetic/mixup-v2-128")
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")

        print(f"=== running mixup for config '{name}' ===")
        run_mixup_inference(
            config_name=name,
            input_pkl=args.input_pkl,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

import argparse

from configs import CONFIGS
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=sorted(CONFIGS.keys()))
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--pkl_path", default="/lustre/nj/cvpr2026/pickles/pca/vindr-v2-128.pkl")
    parser.add_argument("--out_dir", default="weights/vindr-v2-128")
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")

        print(f"=== running config '{name}' ===")
        train(
            config=CONFIGS[name],
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            pkl_path=args.pkl_path,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys

from configs import CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=sorted(CONFIGS.keys()))
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--pkl_path", default="/lustre/nj/cvpr2026/pickles/pca/vindr-v2-128.pkl")
    parser.add_argument("--out_dir", default="weights/vindr-v2-128")
    return parser.parse_args()


def main():
    args = parse_args()

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config: {name}")

        save_dir = os.path.join(args.out_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "train.log")

        print(f"=== running config '{name}' (log: {log_path}) ===")

        with open(log_path, "w") as log_file:
            subprocess.run(
                [
                    sys.executable, "train.py",
                    "--config", name,
                    "--max_steps", str(args.max_steps),
                    "--pkl_path", args.pkl_path,
                    "--out_dir", args.out_dir,
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()

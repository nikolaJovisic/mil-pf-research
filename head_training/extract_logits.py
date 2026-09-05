import argparse
import os
import pickle

import torch

from head_training_ import _train, load_cfg
from utils.evaluation_report import EvaluationReport


def collect_logits(model, dataset, device):
    # Mirrors head_training.utils.evaluate.evaluate()'s loop exactly, but
    # stops short of the sigmoid + EvaluationReport wrapping: raw logits
    # aren't exposed anywhere else in the training/eval pipeline.
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, _, group, instance_type in dataset:
            x, group, instance_type = x.to(device), group.to(device), instance_type.to(device)
            logits = model(x, group, instance_type)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)
    return logits, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default="logit_extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_cfg()
    cfg.pickle_path = args.pickle_path
    # a fresh classifier trained only on this pickle's train split, exactly
    # like each manual_grid_search.py run -- not fine-tuned from anything,
    # and not persisted anywhere else.
    cfg.load_path = None
    cfg.save_path = None

    train_ds, valid_ds, test_ds = pickle.load(open(cfg.pickle_path, "rb"))

    log_dir = os.path.join(cfg.logs_path, args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.csv")

    model = _train(train_ds, valid_ds, cfg, device, log_file, just_evaluate=False, run_id=args.run_id)

    logits, labels = collect_logits(model, test_ds, device)

    # same specificity-at-90%-sensitivity operating point as Table 5, so the
    # plotting side can mark where this run's own decision threshold falls
    # in logit space.
    report = EvaluationReport(torch.sigmoid(logits), labels)
    spec_90 = report.specificity_at(0.9)
    prob_threshold = next(
        (t for t in torch.linspace(1, 0, EvaluationReport.LINSPACE_STEPS).tolist() if report.sensitivity(t) >= 0.9),
        0.5,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "logits": logits,
            "labels": labels,
            # plain floats, not the numpy scalars sklearn hands back, so the
            # file stays loadable under torch.load's weights_only default
            "auc": float(report.auc()),
            "spec_90": float(spec_90),
            "prob_threshold_90": float(prob_threshold),
        },
        out_path,
    )
    print(f"Saved {logits.numel()} test-set logits to {out_path} (AUC={report.auc():.3f}, Spec@90={spec_90:.3f})")


if __name__ == "__main__":
    main()

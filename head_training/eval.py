import torch
import uuid
import csv
import os
from icecream import ic

from head_training_ import load_cfg, train_head

cfg = load_cfg()

test_report, valid_report, train_report = train_head(
    'eval_only',
    cfg,
    just_evaluate=True
)

summary_keys = test_report.summary_keys()
test_summary = test_report.summary()

test_probs = test_report.probs
test_targs = test_report.targets

# import pandas as pd

# pd.DataFrame({"test_targs": test_targs, "test_probs": test_probs}).to_csv("./embed_msl_results.csv", index=False)


# valid_summary = valid_report.summary()
# train_summary = train_report.summary()

ic({
    **{k: test_summary[k] for k in summary_keys}
})
# ic({
#     **{k: valid_summary[k] for k in summary_keys}
# })
# ic({
#     **{k: train_summary[k] for k in summary_keys}
# })
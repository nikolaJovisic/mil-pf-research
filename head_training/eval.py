import torch
import uuid
import csv
import os
from icecream import ic

from head_training_ import load_cfg, train_head

cfg = load_cfg()

report = train_head(
    'eval_only',
    cfg,
    just_evaluate=True
)

summary = report.summary()
summary_keys = report.summary_keys()

ic({
    **{k: summary[k] for k in summary_keys}
})

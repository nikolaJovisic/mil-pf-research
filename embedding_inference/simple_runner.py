import sys
sys.path.append("..")
from shim import *
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

def run(split):
    ds = MammoDataset(
        DatasetEnum.EMBED,
        return_mode=ReturnMode.BREAST_LABEL,
        labels=[1, 4, 5, 6],
        convert_to=ConvertTo.RGB_TENSOR,
        split=split,
        final_resize=512
    )

    cfg = get_embedding_cfg()
    cfg.run_name = cfg.run_name + f'-{split}'
    inference = EmbeddingInference(ds, cfg) 
    inference.run_images()


for split in ['train', 'valid', 'test']:
    run(split)

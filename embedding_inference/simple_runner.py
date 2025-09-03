import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from omegaconf import OmegaConf
from mammo_datasets import *
from embedding_inference_ import EmbeddingInference

def run(split):
    ds = MammoDataset(
        DatasetEnum.EMBED,
        return_mode=ReturnMode.BREAST_LABEL,
        labels=[1, 4, 5, 6],
        convert_to=ConvertTo.RGB_TENSOR,
        split=split,
        final_resize=2048
    )

    cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    cfg.run_name = split
    inference = EmbeddingInference(ds, cfg) 
    inference.run_images()


for split in ['train', 'valid', 'test']:
    run(split)

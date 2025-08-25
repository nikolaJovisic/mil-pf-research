import yaml
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

def save_embedding_inference(obj, save_path, extra_info=None):
    log = {}

#     dataset_dict = obj.dataset.to_dict()
#     log['dataset'] = dataset_dict['dataset']

    inference_cfg_dict = OmegaConf.to_container(obj.cfg, resolve=True)
    log['inference'] = inference_cfg_dict

    timestamp = (datetime.utcnow() + timedelta(hours=2)).isoformat()
    log['meta'] = {
        'timestamp': timestamp,
    }

    if extra_info is not None:
        log['meta'].update(extra_info)

    with open(save_path, 'w') as f:
        yaml.dump(log, f, sort_keys=False)




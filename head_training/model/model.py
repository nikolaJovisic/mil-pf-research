from head_training.model.cls_architectures import Baseline, Perceiver
from head_training.model.last_state_architectures import Dinout

def build_model(cfg, device):
    return {
        'baseline': Baseline,
        'perceiver': Perceiver,
        'dinout': Dinout
    }[cfg.model_type](cfg).to(device)
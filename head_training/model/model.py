from head_training.model.cls_architectures import Baseline, Perceiver
from head_training.model.last_state_architectures import Dinout, Velo

def build_model(cfg, device):
    return {
        'baseline': Baseline,
        'perceiver': Perceiver,
        'dinout': Dinout,
        'velo': Velo
    }[cfg.model_type](cfg).to(device)
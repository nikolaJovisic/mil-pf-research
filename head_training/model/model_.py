from model.cls_architectures import Baseline, Perceiver
from model.last_state_architectures import Dinout, Velo
from model.trex import Trex

def build_model(cfg, device):
    return {
        'baseline': Baseline,
        'perceiver': Perceiver,
        'dinout': Dinout,
        'velo': Velo,  
        'trex': Trex
    }[cfg.model_type](cfg).to(device)
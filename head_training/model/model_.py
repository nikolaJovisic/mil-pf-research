from head_training.model.cls_architectures import Baseline, Perceiver
from head_training.model.last_state_architectures import Dinout, Velo
from head_training.model.trex import Trex
from head_training.model.trex_global import TrexGlobal
from head_training.model.trex_local import TrexLocal
from head_training.model.trex_both_agg import TrexBothAgg
from head_training.model.linear_probe import LinearProbe

def build_model(cfg, device):
    return {
        'baseline': Baseline,
        'perceiver': Perceiver,
        'dinout': Dinout,
        'velo': Velo,  
        'trex': Trex,   
        'linear_probe': LinearProbe,
        'trex_global': TrexGlobal,
        'trex_local': TrexLocal,
        'trex_both_agg': TrexBothAgg
    }[cfg.model_type](cfg).to(device)
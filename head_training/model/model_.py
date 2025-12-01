from model.cls_architectures import Baseline, Perceiver
from model.last_state_architectures import Dinout, Velo
from model.trex import Trex
from model.trex_global import TrexGlobal
from model.trex_local import TrexLocal
from model.trex_both_agg import TrexBothAgg
from model.linear_probe import LinearProbe

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
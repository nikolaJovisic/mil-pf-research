from dataclasses import dataclass, replace
from typing import Literal


@dataclass
class SetFlowConfig:
    name: str = "baseline"

    # dims
    dim: int = 128
    hidden: int = 512
    attn_dim: int = 32
    cond_dim: int = 16
    n_classes: int = 2
    n_types: int = 2
    num_inducing: int = 4

    # branch ablation: which of the two per-token branches (marginal
    # Token MLP vs. set-interaction ISAB) are used, or a mean-pool
    # alternative to ISAB, or neither (pure conditioning, no set/token
    # processing at all).
    branch_mode: Literal["both", "mlp_only", "isab_only", "pool_only", "none"] = "both"

    # number of Linear+ELU layers in the Token MLP branch. 0 = identity
    # (no marginal modeling beyond the shared projection layers).
    token_mlp_depth: int = 3

    # whether the ISAB branch keeps its residual+LayerNorm around the
    # attention output (norm(h + isab(h))) or is used raw (isab(h)),
    # matching how the Token MLP branch is treated.
    isab_residual: bool = True

    # whether the network re-conditions (FiLM) on the summed branch
    # output before the final projection, or just normalizes+activates.
    double_film: bool = True

    # conditioning ablation
    use_class_cond: bool = True
    use_stream_cond: bool = True

    # training / sampling dynamics
    stochastic_bridge: bool = False
    sigma: float = 0.1
    integrator: Literal["rk2", "euler"] = "rk2"
    num_steps: int = 200
    mix_mode: Literal["NONE", "BAG", "GLOBAL"] = "NONE"
    soft_mix: bool = False


def _variant(base: SetFlowConfig, name: str, **overrides) -> SetFlowConfig:
    return replace(base, name=name, **overrides)


_baseline = SetFlowConfig(name="baseline")

_configs_list = [
    #_baseline,

    # --- branch ablations ---
    _variant(_baseline, "mlp_only", branch_mode="mlp_only"),
    _variant(_baseline, "isab_only", branch_mode="isab_only"),
    _variant(_baseline, "pool_only", branch_mode="pool_only"),
    #_variant(_baseline, "no_branches", branch_mode="none"),

    ## --- token mlp depth sweep ---
    #_variant(_baseline, "token_mlp_depth_0", token_mlp_depth=0),
    _variant(_baseline, "token_mlp_depth_1", token_mlp_depth=1),
    #_variant(_baseline, "token_mlp_depth_2", token_mlp_depth=2),
    _variant(_baseline, "token_mlp_depth_5", token_mlp_depth=5),
    #_variant(_baseline, "token_mlp_depth_7", token_mlp_depth=7),

    ## --- conditioning ablations ---
    #_variant(_baseline, "no_class_cond", use_class_cond=False),
    _variant(_baseline, "no_stream_cond", use_stream_cond=False),
    #_variant(_baseline, "no_cond", use_class_cond=False, use_stream_cond=False),

    ## --- conditioning / architecture mechanism ---
    _variant(_baseline, "single_film", double_film=False),
    #_variant(_baseline, "no_isab_residual", isab_residual=False),

    ## --- ISAB internals ---
    _variant(_baseline, "num_inducing_1", num_inducing=1),
    #_variant(_baseline, "num_inducing_2", num_inducing=2),
    _variant(_baseline, "num_inducing_8", num_inducing=8),
    #_variant(_baseline, "num_inducing_16", num_inducing=16),
    #_variant(_baseline, "attn_dim_16", attn_dim=16),
    #_variant(_baseline, "attn_dim_64", attn_dim=64),

    ## --- capacity ---
    #_variant(_baseline, "hidden_256", hidden=256),
    #_variant(_baseline, "hidden_1024", hidden=1024),
    _variant(_baseline, "cond_dim_8", cond_dim=8),
    _variant(_baseline, "cond_dim_32", cond_dim=32),

    ## --- training / sampling dynamics ---
    #_variant(_baseline, "stochastic_bridge", stochastic_bridge=True),
    #_variant(_baseline, "euler_integrator", integrator="euler"),
    #_variant(_baseline, "num_steps_50", num_steps=50),

    ## --- data mixing (already implemented in dataset.py, unused by default) ---
    #_variant(_baseline, "mix_bag", mix_mode="BAG"),
    #_variant(_baseline, "mix_global", mix_mode="GLOBAL"),
    #_variant(_baseline, "mix_global_soft", mix_mode="GLOBAL", soft_mix=True),
]

CONFIGS = {c.name: c for c in _configs_list}

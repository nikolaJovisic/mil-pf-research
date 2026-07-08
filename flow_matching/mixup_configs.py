from dataclasses import dataclass, replace
from typing import Literal


@dataclass
class MixupConfig:
    name: str = "intra_scale_bag"

    # who a token is allowed to be mixed with: another instance from the
    # same bag, or any instance sharing the bag's class label
    pairing_scope: Literal["bag", "class"] = "bag"

    # restrict the partner to instances of the same stream (local/global);
    # if False, local and global tokens can be mixed with each other
    intra_scale: bool = True

    alpha_mean: float = 0.5
    alpha_std: float = 0.15
    alpha_min: float = 0.2
    alpha_max: float = 0.8


def _variant(base: MixupConfig, name: str, **overrides) -> MixupConfig:
    return replace(base, name=name, **overrides)


_base = MixupConfig(name="intra_scale_bag", pairing_scope="bag", intra_scale=True)

_configs_list = [
    _base,
    _variant(_base, "intra_scale_class", pairing_scope="class"),
    _variant(_base, "cross_scale_bag", intra_scale=False),
    _variant(_base, "cross_scale_class", pairing_scope="class", intra_scale=False),
]

CONFIGS = {c.name: c for c in _configs_list}

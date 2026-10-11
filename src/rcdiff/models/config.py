from __future__ import annotations
import inspect
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import torch
import yaml

from rcdiff.models.random_params.random_weight import RandomWeight
from rcdiff.models.random_params.esn_params import ParametersESN
from rcdiff.models.encoders.esn import ESN
from rcdiff.models.denoisers.simple_mlp import SimpleMLP
# from rcdiff.models.denoisers.timegrad import TimeGradDenoiser


def _from_dict(cls, d: dict):
    """Recursively build a dataclass from a dict; unknown keys raise (catches typos in YAML)."""
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kw = {k: _from_dict(hints[k], v) if is_dataclass(hints[k]) else v for k, v in d.items()}
    return cls(**kw)


@dataclass
class RandomWeightConfig:
    dist: str = "normal"
    sparsity: float = 0.0
    norm_type: Optional[str] = None

@dataclass
class ESNConfig:
    reservoir_size: int = 50
    A: RandomWeightConfig = field(default_factory=lambda: RandomWeightConfig("normal", 0.2, "spectral_radius"))
    C: RandomWeightConfig = field(default_factory=lambda: RandomWeightConfig("uniform", 0.2, "l2"))
    zeta: RandomWeightConfig = field(default_factory=lambda: RandomWeightConfig("uniform", 0.2, "euclidean"))
    leak_rate: float = 0.2
    rho: float = 0.1
    gamma: float = 0.4
    omega: float = 0.0
    activ: str = "tanh"
    seed: int = 12345


@dataclass
class DenoiserConfig:
    name: str = "simple_mlp"
    kwargs: dict = field(default_factory=dict)      # e.g. {"dim_hids": [32, 64, 32]}


@dataclass
class DiffusionConfig:
    num_timesteps: int = 100 


@dataclass
class TrainConfig:
    t0: int = 100                              
    batch_size: int = 64
    lr: float = 1e-3
    max_iters: int = 10_000
    val_fraction: float = 0.2  
    eval_every: int = 100
    patience: int = 20     
    ema_decay: float = 0.999  
    grad_clip: Optional[float] = 1.0
    seed: int = 0
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    name: str = "exp"
    in_dim: int = 3
    esn: ESNConfig = field(default_factory=ESNConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        assert 0.0 <= self.train.val_fraction < 1.0
        assert 0.0 <= self.train.ema_decay < 1.0

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    @classmethod
    def from_yaml(cls, path) -> "ExperimentConfig":
        return _from_dict(cls, yaml.safe_load(Path(path).read_text()))



DENOISERS = {"simple_mlp": SimpleMLP}       


def build_esn(cfg: ExperimentConfig) -> ESN:
    e = cfg.esn
    params = ParametersESN(
        A_RW=RandomWeight(**asdict(e.A)), C_RW=RandomWeight(**asdict(e.C)),
        zeta_RW=RandomWeight(**asdict(e.zeta)),
        leak_rate=e.leak_rate, rho=e.rho, gamma=e.gamma, omega=e.omega, seed=e.seed)
    return ESN(in_dim=cfg.in_dim, reservoir_size=e.reservoir_size, params=params, activ=e.activ)


def build_denoiser(cfg: ExperimentConfig) -> torch.nn.Module:
    cls = DENOISERS[cfg.denoiser.name]
    kw = dict(cfg.denoiser.kwargs, in_dim=cfg.in_dim, cond_dim=cfg.esn.reservoir_size)
    if "num_timesteps" in inspect.signature(cls.__init__).parameters:   # derived, not stored twice
        kw["num_timesteps"] = cfg.diffusion.num_timesteps
    return cls(**kw)





def save_checkpoint(path, cfg, denoiser, esn, ema_state=None, extra=None):
    path = Path(path)
    torch.save({
        "config": asdict(cfg),
        "denoiser": denoiser.state_dict(),
        "ema": ema_state,
        "esn": esn.state_dict(),                    # A, C, zeta buffers
        "torch_version": torch.__version__,
        "extra": extra or {},                       # e.g. iteration, best val loss
    }, path)
    cfg.to_yaml(path.with_suffix(".yaml"))          # readable copy next to the weights


def load_checkpoint(path, use_ema=True):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = _from_dict(ExperimentConfig, ck["config"])
    esn = build_esn(cfg)
    regenerated = {k: v.clone() for k, v in esn.state_dict().items()}
    esn.load_state_dict(ck["esn"])
    if not all(torch.allclose(regenerated[k], ck["esn"][k]) for k in regenerated):
        print("Warning: ESN weights regenerated from the seed differ from the saved ones "
              "(different torch version?). Using the saved ones.")
    net = build_denoiser(cfg)
    net.load_state_dict(ck["ema"] if (use_ema and ck["ema"] is not None) else ck["denoiser"])
    return cfg, esn, net
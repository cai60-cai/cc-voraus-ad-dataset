from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class DeepSetsConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    phi_dim: int = 128
    rho_dim: int = 128
    num_layers: int = 3
    num_sensors: int = 130

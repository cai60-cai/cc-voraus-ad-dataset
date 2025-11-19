from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class MLPImputeConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    hidden_dim: int = 256
    num_layers: int = 3
    num_sensors: int = 130

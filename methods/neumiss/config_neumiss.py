from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class NeuMissConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    hidden_dim: int = 128
    num_layers: int = 4
    neumann_order: int = 3
    num_sensors: int = 130

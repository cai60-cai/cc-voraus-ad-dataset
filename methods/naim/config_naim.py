from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class NAIMConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    embed_dim: int = 160
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    num_sensors: int = 130

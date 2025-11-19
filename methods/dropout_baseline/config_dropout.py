from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class DropoutBaselineConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    hidden_dim: int = 256
    num_layers: int = 3
    feature_dropout: float = 0.2
    channel_dropout: float = 0.1
    num_sensors: int = 130

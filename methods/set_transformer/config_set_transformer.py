from __future__ import annotations

from dataclasses import dataclass

from common.config import DatasetConfig, TrainingConfig


@dataclass
class SetTransformerConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    d_phi: int = 128
    num_sensors: int = 130
    num_sab_layers: int = 2
    num_heads: int = 4
    pma_seeds: int = 1

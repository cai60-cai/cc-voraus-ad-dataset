"""Shared configuration dataclasses used throughout the benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    parquet_path: str
    sequence_length: int = 1024
    num_workers: int = 4
    batch_size: int = 8


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    mixed_precision: bool = True
    output_dir: str = "outputs"
    eval_every: int = 1
    checkpoint_name: str = "model.pt"
    seed: int = 42

    def checkpoint_path(self, method_name: str) -> Path:
        out_dir = Path(self.output_dir) / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / self.checkpoint_name

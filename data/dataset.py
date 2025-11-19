"""Data loading utilities for the voraus-AD dataset.

This module centralizes every dataset-specific assumption so that the
model-specific training scripts can share the same interface.  The code
is designed for the benchmark described in the README: we train on
normal samples only and evaluate on normal + anomaly samples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Columns that must never be used for model training.
_METADATA_COLUMNS: Sequence[str] = (
    "time",
    "sample",
    "anomaly",
    "category",
    "setting",
    "action",
    "active",
)


def _load_frame(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Could not find parquet file at {parquet_path}. "
            "Please extract the official voraus-AD dataset."
        )
    return pd.read_parquet(parquet_path)


@dataclass
class NormalizationStats:
    mean: Dict[str, float]
    std: Dict[str, float]

    def save(self, path: Path) -> None:
        payload = {"mean": self.mean, "std": self.std}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @staticmethod
    def load(path: Path) -> "NormalizationStats":
        payload = json.loads(path.read_text())
        return NormalizationStats(mean=payload["mean"], std=payload["std"])


class VorausADDataset(Dataset):
    """PyTorch ``Dataset`` that yields (x, mask, label) dictionaries.

    Each item corresponds to a full robot cycle.  A cycle is identified by
    the ``sample`` field provided in the official dataset.  Because the
    labels are weak, we mark a sequence as anomalous if *any* timestep in
    the cycle is labeled as anomalous.
    """

    def __init__(
        self,
        parquet_path: str,
        split: str = "train",
        sequence_length: int = 1024,
        normalization: str = "zscore",
        stats: Optional[NormalizationStats] = None,
        drop_metadata: bool = True,
        min_timesteps: int = 16,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.split = split
        self.min_timesteps = min_timesteps

        frame = _load_frame(Path(parquet_path))
        frame = frame.sort_values(["sample", "time"]).reset_index(drop=True)
        if split == "train":
            frame = frame[frame["anomaly"] == 0].reset_index(drop=True)

        self.metadata_columns = list(_METADATA_COLUMNS) if drop_metadata else []
        self.feature_columns = [
            col for col in frame.columns if col not in self.metadata_columns
        ]
        if not self.feature_columns:
            raise RuntimeError("No feature columns remaining after filtering metadata.")

        if normalization == "zscore":
            if stats is None:
                stats = self._compute_stats(frame)
            self.stats = stats
            frame = self._apply_normalization(frame, stats)
        else:
            self.stats = None

        self.features = frame[self.feature_columns].to_numpy(np.float32)
        self.labels = frame["anomaly"].to_numpy(np.int64)
        self.sample_ids = frame["sample"].to_numpy(np.int64)

        # Pre-compute indices for each sample id to allow fast slicing without
        # storing duplicated data.
        self.sample_index: List[np.ndarray] = []
        self.sample_labels: List[int] = []
        for sample_id, indices in frame.groupby("sample").indices.items():
            idx = np.asarray(indices)
            if len(idx) < self.min_timesteps:
                continue
            self.sample_index.append(idx)
            label = int(self.labels[idx].max())
            self.sample_labels.append(label)

        if not self.sample_index:
            raise RuntimeError(
                "No samples remaining after filtering. Check dataset path and split."
            )

    def _compute_stats(self, frame: pd.DataFrame) -> NormalizationStats:
        mean = frame[self.feature_columns].mean().to_dict()
        std = frame[self.feature_columns].std().replace(0.0, 1.0).to_dict()
        return NormalizationStats(mean=mean, std=std)

    def _apply_normalization(
        self, frame: pd.DataFrame, stats: NormalizationStats
    ) -> pd.DataFrame:
        for col in self.feature_columns:
            frame[col] = (frame[col] - stats.mean[col]) / stats.std[col]
        return frame

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample_indices = self.sample_index[idx]
        values = self.features[sample_indices]
        label = self.sample_labels[idx]

        # Pad or truncate to the configured sequence length.
        seq = self._pad_or_truncate(values)
        obs_mask = self._build_mask(values)
        obs_mask = self._pad_or_truncate(obs_mask)

        return {
            "x": torch.from_numpy(seq),
            "mask": torch.from_numpy(obs_mask),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _pad_or_truncate(self, array: np.ndarray) -> np.ndarray:
        length = array.shape[0]
        if length >= self.sequence_length:
            return array[: self.sequence_length]
        pad_width = self.sequence_length - length
        pad_shape = (pad_width, array.shape[1])
        pad_values = np.zeros(pad_shape, dtype=array.dtype)
        return np.concatenate([array, pad_values], axis=0)

    def _build_mask(self, values: np.ndarray) -> np.ndarray:
        # Missing sensors are encoded as entire-zero channels.  We mark positions
        # whose absolute value is extremely small as unobserved.
        mask = (np.abs(values) > 1e-6).astype(np.float32)
        return mask

    def get_normalization_stats(self) -> Optional[NormalizationStats]:
        return self.stats


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([item["x"] for item in batch], dim=0)
    mask = torch.stack([item["mask"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {"x": x, "mask": mask, "label": labels}

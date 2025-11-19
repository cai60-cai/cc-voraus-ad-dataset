from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def compute_feature_statistics(values: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes per-feature mean for imputation."""
    obs = (values * mask).sum(axis=0)
    counts = mask.sum(axis=0)
    mean = np.divide(obs, np.maximum(counts, 1), dtype=np.float32)
    return {"mean": mean}


def impute(values: torch.Tensor, mask: torch.Tensor, stats: Dict[str, np.ndarray]) -> torch.Tensor:
    mean = torch.as_tensor(stats["mean"], device=values.device, dtype=values.dtype)
    mean = mean.unsqueeze(0).expand(values.size(0), -1)
    return torch.where(mask > 0, values, mean)

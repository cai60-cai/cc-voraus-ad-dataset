from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead
from .imputation import impute


class MLPImputeModel(nn.Module):
    def __init__(self, config, stats) -> None:
        super().__init__()
        layers = []
        dim = config.num_sensors
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.ReLU())
            dim = config.hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = DeepSVDDHead(config.hidden_dim)
        self.stats = stats

    def _aggregate(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return obs / counts, (counts > 0).float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, sensor_mask = self._aggregate(x, mask)
        imputed = impute(features, sensor_mask, self.stats)
        embedding = self.mlp(imputed)
        return embedding

    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(x, mask)
        return self.head(embedding)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.forward(batch["x"], batch["mask"])
        return self.head.loss(embedding)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.anomaly_score(batch["x"], batch["mask"])
        return scores, batch["label"].float()

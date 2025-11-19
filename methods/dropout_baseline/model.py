from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead


class FeatureDropoutModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        layers = []
        dim = config.num_sensors
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.feature_dropout))
            dim = config.hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = DeepSVDDHead(config.hidden_dim)
        self.channel_dropout = config.channel_dropout

    def _aggregate(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return obs / counts, (counts > 0).float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training and self.channel_dropout > 0.0:
            drop_prob = torch.full_like(mask[:, 0, :], self.channel_dropout)
            channel_keep = torch.bernoulli(1 - drop_prob)
            mask = mask * channel_keep[:, None, :]
        features, _ = self._aggregate(x, mask)
        embedding = self.mlp(features)
        return embedding

    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(x, mask, training=False)
        return self.head(embedding)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.forward(batch["x"], batch["mask"], training=True)
        return self.head.loss(embedding)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.anomaly_score(batch["x"], batch["mask"])
        return scores, batch["label"].float()

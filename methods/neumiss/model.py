from __future__ import annotations

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead


class NeuMissLayer(nn.Module):
    def __init__(self, dim: int, neumann_order: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.neumann_order = neumann_order
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = mask * x
        h = torch.matmul(observed, self.weight) + self.bias
        missing = 1.0 - mask
        correction = torch.zeros_like(h)
        residual = missing * h
        for _ in range(self.neumann_order):
            correction = correction + residual
            residual = self.alpha * missing * residual
        return torch.relu(h + correction)


class NeuMissModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [NeuMissLayer(config.num_sensors, config.neumann_order) for _ in range(config.num_layers)]
        )
        self.projector = nn.Sequential(
            nn.Linear(config.num_sensors, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.head = DeepSVDDHead(config.hidden_dim)

    def _aggregate(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = obs / counts
        sensor_mask = (counts > 0).float()
        return mean, sensor_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        agg, sensor_mask = self._aggregate(x, mask)
        h = agg
        for layer in self.layers:
            h = layer(h, sensor_mask)
        embedding = self.projector(h)
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

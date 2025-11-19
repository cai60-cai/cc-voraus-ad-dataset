from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead, TemporalSensorEncoder


class DeepSets(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TemporalSensorEncoder(out_dim=config.phi_dim)
        rho_layers = []
        dim = config.phi_dim
        for _ in range(config.num_layers):
            rho_layers.append(nn.Linear(dim, config.rho_dim))
            rho_layers.append(nn.ReLU())
            dim = config.rho_dim
        self.rho = nn.Sequential(*rho_layers)
        self.head = DeepSVDDHead(config.rho_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x, mask)
        valid = (mask.sum(dim=1, keepdim=True) > 0).float()
        pooled = (tokens * valid).sum(dim=1)
        embedding = self.rho(pooled)
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

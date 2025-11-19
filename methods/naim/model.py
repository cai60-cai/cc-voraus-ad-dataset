from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead, FeatureTokenizer, TemporalSensorEncoder


class MaskedTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b = tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        mask = torch.cat([torch.zeros(b, 1, device=key_padding_mask.device, dtype=key_padding_mask.dtype), key_padding_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        return encoded[:, 0]


class NAIMModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TemporalSensorEncoder(out_dim=config.embed_dim)
        self.tokenizer = FeatureTokenizer(config.num_sensors, config.embed_dim)
        self.transformer = MaskedTransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.head = DeepSVDDHead(config.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x, mask)
        tokens = self.tokenizer(tokens)
        sensor_missing = mask.sum(dim=1) <= 0
        embedding = self.transformer(tokens, sensor_missing)
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

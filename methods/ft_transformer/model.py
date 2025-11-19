from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead, FeatureTokenizer, TemporalSensorEncoder


class FTTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TemporalSensorEncoder(out_dim=config.embed_dim)
        self.tokenizer = FeatureTokenizer(config.num_sensors, config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.head = DeepSVDDHead(config.embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x, mask)
        tokens = self.tokenizer(tokens)
        b = tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        sensor_missing = mask.sum(dim=1) <= 0
        key_padding = torch.cat(
            [torch.zeros(b, 1, device=tokens.device, dtype=torch.bool), sensor_missing], dim=1
        )
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding)
        return encoded[:, 0]

    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(x, mask)
        return self.head(embedding)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.forward(batch["x"], batch["mask"])
        return self.head.loss(embedding)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.anomaly_score(batch["x"], batch["mask"])
        return scores, batch["label"].float()

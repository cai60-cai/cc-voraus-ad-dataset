from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from common.modules import DeepSVDDHead, FeatureTokenizer, TemporalSensorEncoder


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(queries, keys, keys, key_padding_mask=mask)
        x = self.norm(attn_output + queries)
        return self.ff(x)


class SetAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(dim, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.mab1(x, x, mask)
        return self.mab2(h, h, mask)


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        seed = self.seed.expand(b, -1, -1)
        return self.mab(seed, x, mask)


class SetTransformerModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.sensor_encoder = TemporalSensorEncoder(out_dim=config.d_phi)
        self.tokenizer = FeatureTokenizer(config.num_sensors, config.d_phi)
        self.sab_layers = nn.ModuleList(
            [SetAttentionBlock(config.d_phi, config.num_heads) for _ in range(config.num_sab_layers)]
        )
        self.pma = PoolingByMultiheadAttention(config.d_phi, config.num_heads, config.pma_seeds)
        self.head = DeepSVDDHead(config.d_phi)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.sensor_encoder(x, mask)
        tokens = self.tokenizer(tokens)
        sensor_valid_mask = mask.sum(dim=1) <= 0
        for layer in self.sab_layers:
            tokens = layer(tokens, sensor_valid_mask)
        pooled = self.pma(tokens, sensor_valid_mask).mean(dim=1)
        return pooled

    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(x, mask)
        return self.head(embedding)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.forward(batch["x"], batch["mask"])
        return self.head.loss(embedding)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.anomaly_score(batch["x"], batch["mask"])
        return scores, batch["label"].float()

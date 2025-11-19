"""Reusable neural building blocks."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSensorEncoder(nn.Module):
    """Shared temporal encoder operating on (value, mask) pairs.

    Given ``x`` with shape ``[B, T, S]`` and ``mask`` with the same shape the
    module applies a temporal convolutional network independently to each
    sensor channel and returns ``[B, S, out_dim]`` embeddings.
    """

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = input_channels
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    hidden_channels,
                    kernel_size,
                    padding=((kernel_size - 1) * dilation) // 2,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_ch = hidden_channels
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_channels, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, s = x.shape
        stacked = torch.stack([x, mask], dim=-1)  # [B, T, S, 2]
        stacked = stacked.reshape(b * s, t, 2).permute(0, 2, 1)  # [B*S, 2, T]
        features = self.tcn(stacked)  # [B*S, C, T]
        pooled = features.mean(dim=-1)
        pooled = pooled.view(b, s, -1)
        return self.proj(pooled)


class FeatureTokenizer(nn.Module):
    """Tokenizes per-sensor embeddings with learnable sensor-id embeddings."""

    def __init__(self, num_sensors: int, embed_dim: int) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(num_sensors, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s, d = tokens.shape
        sensor_ids = torch.arange(s, device=tokens.device)
        return tokens + self.id_embedding(sensor_ids)[None, :, :]


class DeepSVDDHead(nn.Module):
    """Distance-to-center anomaly scoring head."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        diff = embedding - self.center
        return torch.sum(diff * diff, dim=-1)

    def loss(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum((embedding - self.center) ** 2, dim=-1))


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + residual)

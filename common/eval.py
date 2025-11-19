"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            batch_scores, batch_labels = model.eval_step({"x": x, "mask": mask, "label": label})
            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    scores_tensor = torch.cat(scores).view(-1)
    labels_tensor = torch.cat(labels).view(-1)
    score_np = scores_tensor.numpy()
    label_np = labels_tensor.numpy()

    metrics = {"auc": float("nan"), "pr": float("nan")}
    if len(np.unique(label_np)) > 1:
        metrics["auc"] = float(roc_auc_score(label_np, score_np))
        metrics["pr"] = float(average_precision_score(label_np, score_np))
    return metrics

from __future__ import annotations

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from common.config import DatasetConfig, TrainingConfig
from common.train_loop import Trainer
from common.utils import seed_everything
from data.dataset import VorausADDataset, collate_fn

from .config_mlp_impute import MLPImputeConfig
from .model import MLPImputeModel


def compute_stats(dataset: VorausADDataset):
    total = None
    counts = None
    for sample in dataset:
        x = sample["x"].numpy()
        mask = sample["mask"].numpy()
        obs = (x * mask).sum(axis=0)
        cnt = mask.sum(axis=0)
        if total is None:
            total = obs
            counts = cnt
        else:
            total += obs
            counts += cnt
    mean = total / np.maximum(counts, 1)
    return {"mean": mean.astype(np.float32)}


def build_dataloaders(config: MLPImputeConfig):
    train_dataset = VorausADDataset(
        parquet_path=config.dataset.parquet_path,
        split="train",
        sequence_length=config.dataset.sequence_length,
    )
    stats = train_dataset.get_normalization_stats()
    test_dataset = VorausADDataset(
        parquet_path=config.dataset.parquet_path,
        split="test",
        sequence_length=config.dataset.sequence_length,
        stats=stats,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataset, train_loader, test_loader


def main() -> None:
    dataset_cfg = DatasetConfig(parquet_path="data/voraus-ad-dataset-100hz.parquet")
    training_cfg = TrainingConfig(learning_rate=1e-3)
    config = MLPImputeConfig(dataset=dataset_cfg, training=training_cfg)

    seed_everything(config.training.seed)
    train_dataset, train_loader, test_loader = build_dataloaders(config)
    stats = compute_stats(train_dataset)
    model = MLPImputeModel(config, stats)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    trainer = Trainer(model, optimizer, config.training)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()

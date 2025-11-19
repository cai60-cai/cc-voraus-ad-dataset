from __future__ import annotations

from torch.utils.data import DataLoader
import torch.optim as optim

from common.config import DatasetConfig, TrainingConfig
from common.train_loop import Trainer
from common.utils import seed_everything
from data.dataset import VorausADDataset, collate_fn

from .config_set_transformer import SetTransformerConfig
from .model import SetTransformerModel


def build_dataloaders(config: SetTransformerConfig):
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
    return train_loader, test_loader


def main() -> None:
    dataset_cfg = DatasetConfig(parquet_path="data/voraus-ad-dataset-500hz.parquet")
    training_cfg = TrainingConfig()
    config = SetTransformerConfig(dataset=dataset_cfg, training=training_cfg)

    seed_everything(config.training.seed)
    model = SetTransformerModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    train_loader, test_loader = build_dataloaders(config)
    trainer = Trainer(model, optimizer, config.training)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()

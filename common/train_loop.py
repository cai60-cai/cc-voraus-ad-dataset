"""Generic training loop shared by all methods."""
from __future__ import annotations

from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import evaluate
from .utils import get_device, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = get_device()
        self.scaler = GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")
        self.model.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                self.optimizer.zero_grad()
                with autocast(enabled=self.scaler.is_enabled()):
                    loss = self.model.training_step(batch)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

            if epoch % self.config.eval_every == 0:
                metrics = evaluate(self.model, val_loader, self.device)
                tqdm.write(f"Epoch {epoch}: AUC={metrics['auc']:.4f}, PR={metrics['pr']:.4f}")
                ckpt_path = self.config.checkpoint_path(self.model.__class__.__name__)
                save_checkpoint(self.model, self.optimizer, epoch, ckpt_path)

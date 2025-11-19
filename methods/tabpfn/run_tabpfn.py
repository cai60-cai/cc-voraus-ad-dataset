from __future__ import annotations

import argparse
import random
from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

try:
    from tabpfn import TabPFNClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("TabPFN is required for this script. Install via `pip install tabpfn`." ) from exc

from data.dataset import VorausADDataset


def sample_vectors(dataset: VorausADDataset, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    labels = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices[:num_samples]:
        sample = dataset[idx]
        x = sample["x"].numpy()
        mask = sample["mask"].numpy()
        vec = (x * mask).sum(axis=0)
        counts = np.maximum(mask.sum(axis=0), 1)
        vec = vec / counts
        vectors.append(vec)
        labels.append(sample["label"].item())
    return np.stack(vectors), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN on voraus-AD vectors")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=2048)
    args = parser.parse_args()

    train_dataset = VorausADDataset(parquet_path=args.parquet, split="train")
    test_dataset = VorausADDataset(parquet_path=args.parquet, split="test", stats=train_dataset.get_normalization_stats())

    X_train, y_train = sample_vectors(train_dataset, args.train_samples)
    X_test, y_test = sample_vectors(test_dataset, args.test_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier(device=device)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)
        print(f"TabPFN AUC: {auc:.4f} PR: {pr:.4f}")
    else:
        print("Test labels contain a single class; metrics undefined.")


if __name__ == "__main__":
    main()

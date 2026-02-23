from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """Memory-mapped chess dataset for efficient random access.

    Backed by pre-tokenized numpy arrays on disk. The OS page cache handles
    memory management — no manual budgeting required.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        targets: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> None:
        super().__init__()
        self.tokens = tokens
        self.targets = targets
        self.start = start_idx
        self.end = end_idx

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.start + idx
        tokens = torch.from_numpy(self.tokens[i].copy()).long()
        target = torch.tensor(self.targets[i], dtype=torch.long)

        return tokens, target


def load_dataset(
    data_dir: str,
    val_split: float = 0.05,
) -> tuple[ChessDataset, ChessDataset]:
    """Load preprocessed binary dataset with memory-mapped arrays.

    Args:
        data_dir: Directory containing tokens.npy, targets.npy, and metadata.json
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    N = metadata["num_positions"]

    tokens = np.load(os.path.join(data_dir, "tokens.npy"), mmap_mode="r")
    targets = np.load(os.path.join(data_dir, "targets.npy"), mmap_mode="r")

    assert tokens.shape == (N, 65), f"Expected tokens shape ({N}, 65), got {tokens.shape}"
    assert targets.shape == (N,), f"Expected targets shape ({N},), got {targets.shape}"

    n_val = int(N * val_split)
    n_train = N - n_val

    train_dataset = ChessDataset(tokens, targets, 0, n_train)
    val_dataset = ChessDataset(tokens, targets, n_train, N)

    return train_dataset, val_dataset

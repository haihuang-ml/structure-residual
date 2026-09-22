from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        num_samples: int | None = None,
        random_sampling: bool = True,
        seed: int = 1234,
        dtype: str | np.dtype = np.uint16,
    ):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.random_sampling = random_sampling
        self.seed = seed
        self.tokens = np.memmap(self.bin_path, dtype=np.dtype(dtype), mode="r")

        min_tokens = seq_len + 1
        if len(self.tokens) < min_tokens:
            raise ValueError(f"{self.bin_path} has {len(self.tokens)} tokens, need at least {min_tokens}")

        self.max_start = len(self.tokens) - seq_len - 1
        default_samples = max(1, self.max_start // seq_len)
        self.num_samples = int(num_samples) if num_samples is not None else default_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.random_sampling:
            rng = np.random.default_rng(self.seed + int(idx))
            start = int(rng.integers(0, self.max_start + 1))
        else:
            start = int((idx * self.seq_len) % (self.max_start + 1))

        window = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        input_ids = torch.from_numpy(window[:-1].copy()).long()
        labels = torch.from_numpy(window[1:].copy()).long()
        return {"input_ids": input_ids, "labels": labels}

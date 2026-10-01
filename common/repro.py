"""Reproducibility helpers for the multi-seed RGB/IR fusion campaign.

The training scripts previously seeded only the train/val/test split
(`split_indices`). They never seeded weight init, augmentation order, or
cuDNN, so running multiple "seeds" barely changed anything. `set_global_seed`
fixes every RNG source and (optionally) requests deterministic kernels.

Usage in a trainer:

    from repro import set_global_seed, seed_worker, make_generator
    set_global_seed(args.seed)
    ...
    loader = DataLoader(ds, shuffle=True, worker_init_fn=seed_worker,
                        generator=make_generator(args.seed))
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG source used during training.

    Args:
        seed: integer seed.
        deterministic: if True, request deterministic cuDNN/algorithms. We use
            ``warn_only=True`` so ops without a deterministic implementation
            fall back gracefully instead of raising (important because some
            YOLO/torchvision ops lack deterministic kernels).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS determinism for matmul on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch without warn_only — best effort.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that derives a stable per-worker seed.

    PyTorch sets ``initial_seed()`` per worker from the generator; we propagate
    it to numpy/random so augmentations relying on those are deterministic too.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU torch.Generator seeded for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

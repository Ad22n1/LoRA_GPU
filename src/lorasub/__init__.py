"""lorasub — major vs minor singular subspaces for LoRA (format vs facts)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

__version__ = "0.1.0"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices) in one place."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def expand_path(p: str | os.PathLike) -> str:
    """Expand $VARS and ~ in a path string (used for $SHARED in configs)."""
    return os.path.expanduser(os.path.expandvars(str(p)))

"""
Reproducibility utilities for DementiaVoiceAI.

Sets seeds for Python, NumPy, and (optionally) PyTorch to ensure
deterministic experiment results.

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set the random seed for all relevant libraries.

    Parameters
    ----------
    seed : int
        The seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (optional — only if installed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


"""
seed.py

Utility functions for ensuring reproducibility throughout the project.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int, optional
        Random seed value. Default is 42.

    Returns
    -------
    None
    """

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed successfully initialized to {seed}.")

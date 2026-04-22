from __future__ import annotations

import os
import random

import numpy as np
import torch

# By fixing the random number generators across different libraries, you ensure that running the same script multiple times will produce the exact same results
# Essential for debugging and comparing model performance
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    # These set the initial states for Python's built-in random module and NumPy’s random generator 
    # Ensuring consistent data shuffling, sampling, or initialization handled by those libraries
    random.seed(seed)
    np.random.seed(seed)
    # Sets the seed for CPU operations in PyTorch
    torch.manual_seed(seed)
    # Sets the seed for all available GPU devices
    torch.cuda.manual_seed_all(seed)

    # In Python, hash-based data structures (like dictionaries and sets) use a random salt for their hash function by default. This makes the order of elements non-deterministic across runs
    # Setting this environment variable forces a fixed salt, ensuring that dictionary/set operations behave consistently
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Forces PyTorch to use algorithms that are guaranteed to be deterministic
        # If a non-deterministic operation is encountered, it will raise an error
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

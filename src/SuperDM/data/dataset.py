from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from SuperDM.data.preprocessing import TransitionScaler


class DynamicsDataset(Dataset):
    """PyTorch dataset for supervised transition modeling."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        scaler: TransitionScaler,
    ) -> None:
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.next_states = next_states.astype(np.float32)
        self.scaler = scaler

        self.inputs, self.targets = scaler.transform(
            self.states, self.actions, self.next_states
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
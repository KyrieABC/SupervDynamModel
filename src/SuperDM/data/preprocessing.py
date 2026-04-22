from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


class TransitionScaler:
    """
    Standardizes:
    - model inputs: concat(state, action)
    - model targets: next_state
    """

    def __init__(self) -> None:
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        x = np.concatenate([states, actions], axis=1)
        y = next_states

        self.input_scaler.fit(x)
        self.target_scaler.fit(y)
        self.is_fitted = True

    def transform(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        x = np.concatenate([states, actions], axis=1)
        y = next_states
        return (
            self.input_scaler.transform(x).astype(np.float32),
            self.target_scaler.transform(y).astype(np.float32),
        )

    def transform_inputs(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        self._check_fitted()
        x = np.concatenate([states, actions], axis=1)
        return self.input_scaler.transform(x).astype(np.float32)

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.target_scaler.inverse_transform(y_scaled).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TransitionScaler":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("TransitionScaler must be fitted before use.")
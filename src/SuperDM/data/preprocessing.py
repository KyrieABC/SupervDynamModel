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
        # Separate scalar for input (s,a) and target (s')
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        #Ensure that scalar had learned mean and variance before attempt to transform data
        self.is_fitted = False

    # Learns the statistics of the dataset
    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        # Concantenate state and action array horizontally (axis=1)to create a single input feature vector
        x = np.concatenate([states, actions], axis=1)
        y = next_states
        
        # It computes the mean and standard deviation for every dimension in both the concatenated input and the target, storing these inside the respective StandardScalar instance
        self.input_scaler.fit(x)
        self.target_scaler.fit(y)
        self.is_fitted = True

    # Scales both feature and target to be ready to be feed into the neural network
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

    # Used during inference
    # Scale input in the same way that model saw during training
    def transform_inputs(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        self._check_fitted()
        x = np.concatenate([states, actions], axis=1)
        return self.input_scaler.transform(x).astype(np.float32)

    # Neural networks are predicting the next state
    # In order to get a meaningful prediction, you must reverse the scaling
    # This method takes the model's output and return it to its original scale
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.target_scaler.inverse_transform(y_scaled).astype(np.float32)

    # save and load serialize the entire project

    # Once you fit a scaler on your training data, you must save it
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    # When you deploy the model later, you must load the exact same scaler to ensure the live environment data is scaled using the same statistics (mean/variance) calculated from your training set
    @classmethod
    def load(cls, path: str | Path) -> "TransitionScaler":
        with open(path, "rb") as f:
            return pickle.load(f)

    # Defensive programming pattern
    # prevents subtle bugs where an unfitted (empty) scaler would produce mathematically incorrect transformations
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("TransitionScaler must be fitted before use.")

from __future__ import annotations

import torch
from torch import nn

# Standard feed-forward Multi-Layer Perceptron (MLP) architecture 
class MLPDynamicsModel(nn.Module):
    """MLP baseline for predicting next_state from (state, action)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        predict_delta: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # (s,a) are inputs
        self.input_dim = state_dim + action_dim
        self.output_dim = state_dim
        self.predict_delta = predict_delta

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        # bundles these layers into a single container, which automatically handles the forward pass through the entire network
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    # initializes weights to account for the variance reduction caused by ReLU
    # preventing "vanishing" or "exploding" gradients early in training
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

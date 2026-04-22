from __future__ import annotations

import numpy as np
import torch


def rollout_model_vs_env(env, model, scaler, num_steps: int, seed: int = 0):
    """
    Compare:
    - true environment rollout
    - autoregressive model rollout

    Both start from the same initial state and use the same sampled actions.
    """
    env.rng = np.random.default_rng(seed)
    initial_state = env.reset()

    actions = []
    true_states = [initial_state.copy()]
    pred_states = [initial_state.copy()]

    current_true_state = initial_state.copy()
    current_pred_state = initial_state.copy()

    for _ in range(num_steps):
        action = env.sample_action()
        actions.append(action.copy())

        env.state = current_true_state.copy()
        next_true_state = env.step(action)
        true_states.append(next_true_state.copy())
        current_true_state = next_true_state

        x_scaled = scaler.transform_inputs(
            states=current_pred_state[None, :],
            actions=action[None, :],
        )
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(x_tensor).cpu().numpy()

        next_pred_state = scaler.inverse_transform_targets(pred_scaled)[0]
        pred_states.append(next_pred_state.copy())
        current_pred_state = next_pred_state

    return np.asarray(true_states), np.asarray(pred_states)
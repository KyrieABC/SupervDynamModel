from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = ["x", "y", "vx", "vy"]


def plot_loss_curve(train_losses, val_losses, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_component_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.25, s=6)
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True next_state values")
    plt.ylabel("Predicted next_state values")
    plt.title("Predicted vs True Next-State Components")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_state_component_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    timesteps = np.arange(len(y_true))

    plt.figure(figsize=(10, 6))
    for i in range(y_true.shape[1]):
        plt.plot(timesteps, y_true[:, i], label=f"true_{STATE_NAMES[i]}", alpha=0.9)
        plt.plot(timesteps, y_pred[:, i], linestyle="--", label=f"pred_{STATE_NAMES[i]}", alpha=0.9)

    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.title("True vs Predicted State Components")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rollout_xy(rollout_pairs, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    for idx, (true_states, pred_states) in enumerate(rollout_pairs):
        plt.plot(true_states[:, 0], true_states[:, 1], label=f"true_rollout_{idx}")
        plt.plot(pred_states[:, 0], pred_states[:, 1], linestyle="--", label=f"pred_rollout_{idx}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs Predicted XY Rollouts")
    plt.legend(fontsize=8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
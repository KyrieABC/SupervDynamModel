import argparse
from pathlib import Path

import numpy as np
import torch

from SuperDM.data.preprocessing import TransitionScaler
from SuperDM.evaluation.metrics import compute_regression_metrics
from SuperDM.evaluation.rollout import rollout_model_vs_env
from SuperDM.models.mlp import MLPDynamicsModel
from SuperDM.simulation.point_mass import PointMass2DEnv
from SuperDM.utility.io import load_json, ensure_dir, save_json
from SuperDM.utility.plotting import (
    plot_component_scatter,
    plot_rollout_xy,
    plot_state_component_timeseries,
)
from SuperDM.utility.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained dynamics model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to JSON config file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    set_global_seed(config["seed"], deterministic=config.get("deterministic", True))

    eval_dir = Path(config["evaluation"]["output_dir"])
    ensure_dir(eval_dir)

    model_cfg = config["model"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]

    artifacts_dir = Path(training_cfg["artifacts_dir"])
    checkpoint_path = Path(training_cfg["checkpoint_dir"]) / "best.pt"
    scaler_path = artifacts_dir / "scaler.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = TransitionScaler.load(scaler_path)

    model = MLPDynamicsModel(
        state_dim=model_cfg["state_dim"],
        action_dim=model_cfg["action_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
        predict_delta=model_cfg["predict_delta"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = np.load(Path(dataset_cfg["output_path"]))
    states = dataset["states"]
    actions = dataset["actions"]
    next_states = dataset["next_states"]

    x_scaled, y_scaled = scaler.transform(states, actions, next_states)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()

    pred_next_states = scaler.inverse_transform_targets(pred_scaled)
    metrics = compute_regression_metrics(next_states, pred_next_states)
    save_json(metrics, eval_dir / "metrics.json")

    plot_component_scatter(
        y_true=next_states,
        y_pred=pred_next_states,
        save_path=eval_dir / "component_scatter.png",
    )

    plot_state_component_timeseries(
        y_true=next_states[:300],
        y_pred=pred_next_states[:300],
        save_path=eval_dir / "state_component_timeseries.png",
    )

    env = PointMass2DEnv(
        dt=dataset_cfg["dt"],
        damping=dataset_cfg["damping"],
        max_position=dataset_cfg["max_position"],
        max_velocity=dataset_cfg["max_velocity"],
        max_acceleration=dataset_cfg["max_acceleration"],
        process_noise_std=dataset_cfg["process_noise_std"],
        seed=config["seed"],
    )

    rollout_pairs = []
    for rollout_seed in range(eval_cfg["num_rollouts"]):
        true_states, pred_states = rollout_model_vs_env(
            env=env,
            model=model,
            scaler=scaler,
            num_steps=eval_cfg["num_rollout_steps"],
            seed=config["seed"] + rollout_seed,
        )
        rollout_pairs.append((true_states, pred_states))

    plot_rollout_xy(
        rollout_pairs=rollout_pairs,
        save_path=eval_dir / "rollout_xy.png",
    )

    print(f"Saved evaluation metrics to: {eval_dir / 'metrics.json'}")
    print(f"Saved plots to: {eval_dir}")


if __name__ == "__main__":
    main()
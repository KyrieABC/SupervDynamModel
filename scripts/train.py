import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from SuperDM.data.dataset import DynamicsDataset
from SuperDM.data.preprocessing import TransitionScaler
from SuperDM.models.mlp import MLPDynamicsModel
from SuperDM.training.trainer import Trainer
from SuperDM.utility.io import load_json, ensure_dir, save_json
from SuperDM.utility.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised dynamics model.")
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

    dataset_path = Path(config["dataset"]["output_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run data/generate_dataset.py first."
        )

    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    indices = np.arange(len(states))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config["split"]["val_ratio"],
        random_state=config["seed"],
        shuffle=True,
    )

    scaler = TransitionScaler()
    scaler.fit(states[train_idx], actions[train_idx], next_states[train_idx])

    train_dataset = DynamicsDataset(
        states=states[train_idx],
        actions=actions[train_idx],
        next_states=next_states[train_idx],
        scaler=scaler,
    )
    val_dataset = DynamicsDataset(
        states=states[val_idx],
        actions=actions[val_idx],
        next_states=next_states[val_idx],
        scaler=scaler,
    )

    model_cfg = config["model"]
    model = MLPDynamicsModel(
        state_dim=model_cfg["state_dim"],
        action_dim=model_cfg["action_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
        predict_delta=model_cfg["predict_delta"],
    )

    artifacts_dir = Path(config["training"]["artifacts_dir"])
    ensure_dir(artifacts_dir)
    save_json(config, artifacts_dir / "train_config_snapshot.json")
    scaler.save(artifacts_dir / "scaler.pkl")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        scaler=scaler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import numpy as np

from SuperDM.simulation.point_mass import PointMass2DEnv
from SuperDM.utility.io import load_json, ensure_dir
from SuperDM.utility.reproducibility import set_global_seed


def generate_dataset(config: dict) -> None:
    dataset_cfg = config["dataset"]
    seed = config["seed"]

    set_global_seed(seed, deterministic=config.get("deterministic", True))

    env = PointMass2DEnv(
        dt=dataset_cfg["dt"],
        damping=dataset_cfg["damping"],
        max_position=dataset_cfg["max_position"],
        max_velocity=dataset_cfg["max_velocity"],
        max_acceleration=dataset_cfg["max_acceleration"],
        process_noise_std=dataset_cfg["process_noise_std"],
        seed=seed,
    )

    num_episodes = dataset_cfg["num_episodes"]
    steps_per_episode = dataset_cfg["steps_per_episode"]

    states = []
    actions = []
    next_states = []
    episode_ids = []

    for episode_idx in range(num_episodes):
        state = env.reset()
        for _ in range(steps_per_episode):
            action = env.sample_action()
            next_state = env.step(action)

            states.append(state.copy())
            actions.append(action.copy())
            next_states.append(next_state.copy())
            episode_ids.append(episode_idx)

            state = next_state

    output_path = Path(dataset_cfg["output_path"])
    ensure_dir(output_path.parent)

    np.savez_compressed(
        output_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )

    print(f"Saved dataset to: {output_path}")
    print(f"Num transitions: {len(states)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic dynamics dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to JSON config file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_json(args.config)
    generate_dataset(config)
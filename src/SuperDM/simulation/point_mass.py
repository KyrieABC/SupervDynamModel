from __future__ import annotations

import numpy as np


class PointMass2DEnv:
    """
    Simple 2D point-mass dynamics simulator.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Dynamics:
        vx_{t+1} = vx_t + (ax - damping * vx_t) * dt
        vy_{t+1} = vy_t + (ay - damping * vy_t) * dt
        x_{t+1}  = x_t + vx_{t+1} * dt
        y_{t+1}  = y_t + vy_{t+1} * dt
    """

    def __init__(
        self,
        dt: float = 0.1,
        damping: float = 0.08,
        max_position: float = 10.0,
        max_velocity: float = 5.0,
        max_acceleration: float = 1.0,
        process_noise_std: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.dt = dt
        self.damping = damping
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.process_noise_std = process_noise_std
        self.rng = np.random.default_rng(seed)

        self.state_dim = 4
        self.action_dim = 2
        self.state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        x = self.rng.uniform(-1.0, 1.0)
        y = self.rng.uniform(-1.0, 1.0)
        vx = self.rng.uniform(-0.5, 0.5)
        vy = self.rng.uniform(-0.5, 0.5)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        return self.state.copy()

    def sample_action(self) -> np.ndarray:
        action = self.rng.uniform(
            low=-self.max_acceleration,
            high=self.max_acceleration,
            size=(2,),
        )
        return action.astype(np.float32)

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        x, y, vx, vy = self.state
        ax, ay = action

        vx_next = vx + (ax - self.damping * vx) * self.dt
        vy_next = vy + (ay - self.damping * vy) * self.dt

        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt

        next_state = np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)

        if self.process_noise_std > 0.0:
            next_state += self.rng.normal(
                loc=0.0,
                scale=self.process_noise_std,
                size=(self.state_dim,),
            ).astype(np.float32)

        next_state[0:2] = np.clip(next_state[0:2], -self.max_position, self.max_position)
        next_state[2:4] = np.clip(next_state[2:4], -self.max_velocity, self.max_velocity)

        self.state = next_state.astype(np.float32)
        return self.state.copy()
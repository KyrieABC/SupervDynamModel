import numpy as np

from SuperDM.simulation.point_mass import PointMass2DEnv


def test_reset_returns_valid_state_shape():
    env = PointMass2DEnv(seed=123)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_step_returns_valid_state_shape():
    env = PointMass2DEnv(seed=123)
    env.reset()
    action = np.array([0.2, -0.1], dtype=np.float32)
    next_state = env.step(action)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)
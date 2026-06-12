import gymnasium as gym
import numpy as np
import pytest


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.value = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0
        return np.full(self.observation_space.shape, self.value, dtype=np.uint8), {}

    def step(self, action):
        self.value += 1
        obs = np.full(self.observation_space.shape, self.value, dtype=np.uint8)
        return obs, float(action), self.value >= 2, False, {"value": self.value}


def make_counter_env():
    return CounterEnv()


def test_stable_retro_subproc_vec_env_shared_memory():
    pytest.importorskip("stable_baselines3")
    from stable_retro.vec_env import StableRetroSubprocVecEnv

    env = StableRetroSubprocVecEnv(
        [make_counter_env, make_counter_env],
        start_method="fork",
    )
    try:
        obs = env.reset()
        assert obs.shape == (2, 2, 2, 1)
        assert np.all(obs == 0)

        obs, rewards, dones, infos = env.step(np.asarray([0, 1]))
        assert np.all(obs == 1)
        assert rewards.tolist() == [0.0, 1.0]
        assert dones.tolist() == [False, False]
        assert infos == [{"value": 1}, {"value": 1}]

        obs, rewards, dones, infos = env.step(np.asarray([1, 1]))
        assert np.all(obs == 0)
        assert rewards.tolist() == [1.0, 1.0]
        assert dones.tolist() == [True, True]
        assert "terminal_observation" in infos[0]
    finally:
        env.close()

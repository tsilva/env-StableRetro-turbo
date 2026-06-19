import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_vec_env.py"
    spec = importlib.util.spec_from_file_location("benchmark_vec_env", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyImageEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(4, 4, 3),
            dtype=np.uint8,
        )
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros((4, 4, 3), dtype=np.uint8), {"reset": True}

    def step(self, action):
        self.step_count += 1
        obs = np.full((4, 4, 3), self.step_count * 10, dtype=np.uint8)
        return obs, float(self.step_count), False, False, {"step": self.step_count}


def test_auto_backend_falls_back_to_subproc_when_native_missing(monkeypatch):
    bench = _load_benchmark_module()

    monkeypatch.setattr(bench, "_native_vec_available", lambda: False)
    assert bench._resolve_backend("auto") == "subproc"
    assert bench._resolve_backend("native") == "native"

    monkeypatch.setattr(bench, "_native_vec_available", lambda: True)
    assert bench._resolve_backend("auto") == "native"


def test_regular_preprocess_wrapper_matches_profile_shape_and_skip():
    bench = _load_benchmark_module()
    env = DummyImageEnv()

    wrapped = bench.BenchmarkRetroPreprocessWrapper(
        env,
        obs_resize=(2, 2),
        obs_crop=(1, 1, 1, 1),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=2,
        frame_stack=3,
        maxpool_last_two=True,
    )

    obs, info = wrapped.reset()
    assert info == {"reset": True}
    assert obs.shape == (2, 2, 3)
    assert obs.dtype == np.uint8
    assert np.array_equal(obs[:, :, 0], obs[:, :, 1])
    assert np.array_equal(obs[:, :, 1], obs[:, :, 2])

    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert env.step_count == 2
    assert reward == 3.0
    assert terminated is False
    assert truncated is False
    assert info == {"step": 2}
    assert obs.shape == (2, 2, 3)
    assert np.all(obs[:, :, -1] == 20)

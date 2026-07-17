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
    monkeypatch.setattr(bench, "_sb3_vec_available", lambda: True)
    assert bench._resolve_backend("auto") == "subproc"
    assert bench._resolve_backend("native") == "native"

    monkeypatch.setattr(bench, "_sb3_vec_available", lambda: False)
    assert bench._resolve_backend("auto") == "async"

    monkeypatch.setattr(bench, "_native_vec_available", lambda: True)
    assert bench._resolve_backend("auto") == "native"


def test_resolve_game_accepts_short_game_and_platform():
    bench = _load_benchmark_module()

    assert (
        bench._resolve_game("SuperMarioBros-Nes-v0", "MegaMan", "Nes")
        == "MegaMan-Nes-v0"
    )
    assert (
        bench._resolve_game("SuperMarioBros-Nes-v0", "MegaMan-Nes-v0", "Nes")
        == "MegaMan-Nes-v0"
    )
    assert (
        bench._resolve_game("SuperMarioBros-Nes-v0", None, None)
        == "SuperMarioBros-Nes-v0"
    )


def test_resolve_game_rejects_platform_without_matching_game():
    bench = _load_benchmark_module()

    try:
        bench._resolve_game("SuperMarioBros-Nes-v0", None, "Nes")
    except SystemExit as exc:
        assert "--platform requires --game" in str(exc)
    else:
        raise AssertionError("expected --platform without --game to exit")

    try:
        bench._resolve_game("MegaMan-Nes-v0", "MegaMan-Nes-v0", "Snes")
    except SystemExit as exc:
        assert "does not match --platform" in str(exc)
    else:
        raise AssertionError("expected mismatched platform to exit")


def test_regular_preprocess_wrapper_matches_profile_shape_and_skip():
    bench = _load_benchmark_module()
    env = DummyImageEnv()

    wrapped = bench.BenchmarkRetroPreprocessWrapper(
        env,
        obs_resize=(2, 2),
        obs_crop=(1, 1, 1, 1),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_crop_mode="remove",
        obs_crop_fill=0,
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


def test_regular_preprocess_wrapper_supports_mask_crop():
    bench = _load_benchmark_module()
    env = DummyImageEnv()

    wrapped = bench.BenchmarkRetroPreprocessWrapper(
        env,
        obs_resize=None,
        obs_crop=(1, 1, 1, 1),
        obs_grayscale=False,
        obs_resize_algorithm="area",
        obs_crop_mode="mask",
        obs_crop_fill=123,
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
    )

    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    cropped = wrapped._apply_obs_crop(image)
    assert cropped.shape == image.shape
    assert np.all(cropped[0, :, :] == 123)
    assert np.all(cropped[-1, :, :] == 123)
    assert np.all(cropped[:, 0, :] == 123)
    assert np.all(cropped[:, -1, :] == 123)
    np.testing.assert_array_equal(cropped[1:3, 1:3, :], image[1:3, 1:3, :])


def test_named_mario_action_sequence_is_deterministic():
    bench = _load_benchmark_module()

    actions = bench._parse_actions("noop,right,right_b,right_a")
    templates = bench._action_templates(actions, num_envs=2)
    sequence = bench._sample_action_sequence(templates, count=4, seed=0)

    assert sequence[0].shape == (2, 9)
    assert {tuple(batch[0]) for batch in sequence}.issubset(
        {bench.MARIO_SIMPLE_ACTIONS[name] for name in actions}
    )
    sequence_again = bench._sample_action_sequence(templates, count=4, seed=0)
    for left, right in zip(sequence, sequence_again, strict=True):
        np.testing.assert_array_equal(left, right)


def test_dry_run_prints_supermario_canonical_overrides(monkeypatch, capsys):
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench, "_resolve_backend", lambda requested: "native")

    assert (
        bench.main(
            [
                "--profile",
                "supermario-level1-1",
                "--backend",
                "auto",
                "--steps",
                "5",
                "--repeats",
                "3",
                "--warmup-steps",
                "2",
                "--num-envs",
                "4",
                "--num-threads",
                "4",
                "--states",
                "Level1-1,Level1-2",
                "--actions",
                "noop,right",
                "--action-seed",
                "0",
                "--obs-layout",
                "chw",
                "--obs-crop-mode",
                "mask",
                "--no-maxpool-last-two",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "envs=4 threads=4" in output
    assert "state=catalog=Level1-1,Level1-2 default_index=0" in output
    assert "crop_mode=mask" in output
    assert "maxpool_last_two=False" in output
    assert "autoreset_mode=Disabled" in output
    assert "obs_layout=chw" in output
    assert "actions=('noop', 'right')" in output
    assert "steps=5 repeats=3" in output

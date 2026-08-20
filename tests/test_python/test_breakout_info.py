from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _breakout_rom_path_or_skip():
    import env_stableretro_turbo as retro

    try:
        return retro.data.get_original_romfile_path("Breakout-Atari2600-v0")
    except FileNotFoundError:
        pytest.skip("Breakout-Atari2600-v0 ROM is not imported locally")


def test_breakout_lives_info_matches_authority_integration():
    from env_stableretro_turbo.vec_env import RetroVecEnv

    info_path = (
        Path(__file__).resolve().parents[2]
        / "env_stableretro_turbo/data/stable/Breakout-Atari2600-v0/data.json"
    )
    state_path = info_path.with_name("Start.state")
    env = RetroVecEnv(
        "Breakout-Atari2600-v0",
        state=str(state_path),
        info=str(info_path),
        num_envs=1,
        num_threads=1,
        rom_path=_breakout_rom_path_or_skip(),
        obs_copy="copy",
        obs_resize=None,
        obs_grayscale=False,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        frame_skip=4,
        frame_stack=1,
        use_fire_reset=False,
        info_filter={"mode": "all", "keys": ["lives"]},
    )
    try:
        _observations, reset_infos = env.reset(seed=10024)
        assert int(reset_infos["lives"][0]) > 0

        fire = np.zeros((1, env.num_buttons), dtype=np.int8)
        fire[:, 0] = 1
        _observations, _rewards, _terminated, _truncated, step_infos = env.step(fire)

        assert int(step_infos["lives"][0]) == int(reset_infos["lives"][0])
    finally:
        env.close()

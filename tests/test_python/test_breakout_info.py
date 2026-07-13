from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _breakout_rom_path_or_skip():
    import stable_retro as retro

    try:
        return retro.data.get_original_romfile_path("Breakout-Atari2600-v0")
    except FileNotFoundError:
        pytest.skip("Breakout-Atari2600-v0 ROM is not imported locally")


def test_breakout_ball_y_reports_waiting_for_fire_and_active_ball():
    from stable_retro.vec_env import RetroVecEnv

    info_path = (
        Path(__file__).resolve().parents[2]
        / "stable_retro/data/stable/Breakout-Atari2600-v0/data.json"
    )
    env = RetroVecEnv(
        "Breakout-Atari2600-v0",
        state="Start",
        info=str(info_path),
        num_envs=1,
        num_threads=1,
        rom_path=_breakout_rom_path_or_skip(),
        frame_skip=4,
        frame_stack=1,
        use_fire_reset=False,
        info_filter={"mode": "all", "keys": ["ball_y", "lives"]},
        autoreset_mode="Disabled",
    )
    try:
        _observations, reset_infos = env.reset(seed=10024)
        assert int(reset_infos["ball_y"][0]) == 0
        assert int(reset_infos["lives"][0]) > 0

        fire = np.zeros((1, env.num_buttons), dtype=np.int8)
        fire[:, 0] = 1
        _observations, _rewards, _terminated, _truncated, step_infos = env.step(fire)

        assert int(step_infos["ball_y"][0]) > 0
        assert int(step_infos["lives"][0]) == int(reset_infos["lives"][0])
    finally:
        env.close()

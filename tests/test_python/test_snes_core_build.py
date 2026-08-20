from pathlib import Path

import numpy as np
import pytest

import env_stableretro_turbo as retro


def test_snes_core_treats_arm64_as_little_endian():
    port_header = Path(__file__).parents[2] / "cores" / "snes" / "port.h"
    source = port_header.read_text(encoding="utf-8")

    assert "defined(__aarch64__)" in source
    assert "defined(__arm64__)" in source


def test_snes_arm64_core_advances_gameplay_without_false_termination():
    try:
        env = retro.make(game="SuperMarioWorld-Snes-v0", render_mode="rgb_array")
    except FileNotFoundError:
        pytest.skip("Super Mario World ROM is not imported")

    try:
        initial, _ = env.reset()
        action = [0] * len(env.buttons)
        action[env.buttons.index("RIGHT")] = 1
        action[env.buttons.index("B")] = 1

        terminations = 0
        info = {}
        frame = initial
        for _ in range(180):
            frame, _, terminated, truncated, info = env.step(action)
            terminations += int(terminated or truncated)

        assert not np.array_equal(frame, initial)
        assert terminations == 0
        assert info["lives"] > 0
    finally:
        env.close()

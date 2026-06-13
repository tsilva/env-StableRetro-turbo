from pathlib import Path

import numpy as np
import pytest


def test_stable_retro_native_vec_env_same_process():
    pytest.importorskip("stable_baselines3")
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    dummy_json = root / "dummy.json"

    env = StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        4,
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(dummy_json),
        scenario=str(dummy_json),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=2,
        frame_stack=4,
        num_threads=2,
    )
    try:
        obs = env.reset()
        assert obs.shape == (4, 84, 84, 4)
        assert obs.dtype == np.uint8
        assert all(env.observation_space.contains(obs[i]) for i in range(env.num_envs))

        actions = np.asarray([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (4, 84, 84, 4)
        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32
        assert dones.tolist() == [False, False, False, False]
        assert len(infos) == 4
        assert all(env.observation_space.contains(obs[i]) for i in range(env.num_envs))
    finally:
        env.close()

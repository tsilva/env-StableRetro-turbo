import hashlib
from pathlib import Path

import numpy as np
import pytest


def _sha(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ptr(array):
    return int(np.asarray(array).__array_interface__["data"][0])


def _empty_info_path(tmp_path):
    empty_info = tmp_path / "empty_info.json"
    empty_info.write_text('{"info": {}}', encoding="utf-8")
    return empty_info


def _make_test_native_vec_env(tmp_path, **kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    num_envs = kwargs.pop("num_envs", 2)
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    empty_info = _empty_info_path(tmp_path)

    return StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        num_envs,
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(empty_info),
        scenario=str(empty_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=2,
        frame_stack=4,
        num_threads=2,
        **kwargs,
    )


def test_stable_retro_native_vec_env_same_process(tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(
        tmp_path,
        num_envs=4,
        info_mode="terminal",
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


@pytest.mark.parametrize("info_mode", ["terminal", "none"])
def test_stable_retro_native_vec_env_fast_info_modes(info_mode, tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(tmp_path, info_mode=info_mode)
    try:
        env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        _, rewards, dones, infos = env.step(actions)

        assert rewards.tolist() == [0.0, 0.0]
        assert dones.tolist() == [False, False]
        assert infos == [{}, {}]
    finally:
        env.close()


def test_stable_retro_native_vec_env_observations_do_not_alias_previous_step(tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(
        tmp_path,
        copy_observations=False,
        info_mode="terminal",
    )
    try:
        obs0 = env.reset()
        obs0_hash = _sha(obs0)
        obs0_ptr = _ptr(obs0)

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        obs1, _, _, _ = env.step(actions)

        assert _sha(obs0) == obs0_hash
        assert _ptr(obs1) != obs0_ptr
        assert not np.shares_memory(obs1, obs0)
    finally:
        env.close()


def test_stable_retro_native_vec_env_unsafe_zero_copy_aliases_observations(tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(
        tmp_path,
        copy_observations=False,
        unsafe_zero_copy=True,
        info_mode="terminal",
    )
    try:
        obs0 = env.reset()
        obs0_ptr = _ptr(obs0)

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        obs1, _, _, _ = env.step(actions)

        assert _ptr(obs1) == obs0_ptr
        assert np.shares_memory(obs1, obs0)
    finally:
        env.close()


def test_stable_retro_native_vec_env_mario_infos_if_rom_present():
    pytest.importorskip("stable_baselines3")
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    try:
        rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    expected_keys = {
        "coins",
        "levelHi",
        "levelLo",
        "lives",
        "score",
        "scrolling",
        "time",
        "xscrollHi",
        "xscrollLo",
    }

    env = StableRetroNativeVecEnv(
        "SuperMarioBros-Nes-v0",
        1,
        state="Level1-1",
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
    )
    terminal_env = StableRetroNativeVecEnv(
        "SuperMarioBros-Nes-v0",
        1,
        state="Level1-1",
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
        info_mode="terminal",
    )
    try:
        env.reset()
        terminal_env.reset()
        action = np.zeros((1, env.num_buttons), dtype=np.uint8)
        _, rewards, dones, infos = env.step(action)
        _, terminal_rewards, terminal_dones, terminal_infos = terminal_env.step(action)

        assert rewards.tolist() == terminal_rewards.tolist()
        assert dones.tolist() == terminal_dones.tolist()
        assert terminal_infos == [{}]
        assert len(infos) == 1
        assert expected_keys.issubset(infos[0])
        assert all(isinstance(infos[0][key], int) for key in expected_keys)
    finally:
        env.close()
        terminal_env.close()

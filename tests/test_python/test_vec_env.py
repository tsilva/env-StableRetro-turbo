import hashlib
import os
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


def _time_reward_info_path(tmp_path):
    reward_info = tmp_path / "time_reward_info.json"
    reward_info.write_text(
        """
{
  "info": {
    "frame_reward_source": {
      "address": 0,
      "type": "|u1"
    }
  },
  "reward": {
    "time": {
      "measurement": "delta",
      "op": "positive",
      "reference": 0,
      "reward": 1.0,
      "penalty": 0.0
    }
  }
}
""",
        encoding="utf-8",
    )
    return reward_info


def _done_on_frame_info_path(tmp_path):
    done_info = tmp_path / "done_on_frame_info.json"
    done_info.write_text(
        """
{
  "info": {
    "frame_reward_source": {
      "address": 0,
      "type": "|u1"
    }
  },
  "done": {
    "variables": {
      "frame_reward_source": {
        "op": "greater-than",
        "reference": 0
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    return done_info


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


def test_stable_retro_native_vec_env_forwards_per_env_reset_seeds():
    pytest.importorskip("stable_baselines3")
    from stable_retro.vec_env import StableRetroNativeVecEnv

    class FakeNative:
        def __init__(self):
            self.reset_seeds = []

        def reset(self, seed):
            self.reset_seeds.append(seed)
            return np.zeros((3, 2, 2, 1), dtype=np.uint8), [{}, {}, {}]

    env = StableRetroNativeVecEnv.__new__(StableRetroNativeVecEnv)
    env.native = FakeNative()
    env.num_envs = 3
    env.copy_observations = True
    env._seeds = [11, None, 37]
    env._options = [{}, {}, {}]

    obs = env.reset()

    assert obs.shape == (3, 2, 2, 1)
    assert env.native.reset_seeds == [[11, None, 37]]
    assert env._seeds == [None, None, None]


def test_stable_retro_native_vec_env_frame_stack_order(tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(tmp_path, info_mode="terminal")
    try:
        obs = env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        saw_new_frame = False

        for _ in range(8):
            prev = obs.copy()
            obs, _, _, _ = env.step(actions)

            assert np.array_equal(obs[..., :-1], prev[..., 1:])
            saw_new_frame = saw_new_frame or not np.array_equal(
                obs[..., -1],
                prev[..., -1],
            )

        assert saw_new_frame
    finally:
        env.close()


def _native_test_rom_trace(tmp_path, copy_observations, num_threads, seed, actions):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    reward_info = _time_reward_info_path(tmp_path)

    env = StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        8,
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(reward_info),
        scenario=str(reward_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=num_threads,
        copy_observations=copy_observations,
        noop_reset_max=1,
        sticky_action_prob=0.25,
        info_mode="all",
    )
    try:
        env.seed(seed)
        obs = env.reset()
        trace = [(_sha(obs), None, None, None)]
        total_reward = 0.0
        for action in actions:
            obs, rewards, dones, infos = env.step(action)
            total_reward += float(np.sum(rewards))
            trace.append(
                (
                    _sha(obs),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos(infos),
                ),
            )
        assert total_reward > 0.0
        return trace
    finally:
        env.close()


@pytest.mark.parametrize("copy_observations", [True, False])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_stable_retro_native_vec_env_seed_determinism_ci_rom(
    tmp_path,
    copy_observations,
    num_threads,
):
    pytest.importorskip("stable_baselines3")

    actions = np.random.default_rng(20260616).integers(
        0,
        2,
        size=(128, 8, 9),
        dtype=np.uint8,
    )

    first = _native_test_rom_trace(
        tmp_path,
        copy_observations,
        num_threads,
        123,
        actions,
    )
    second = _native_test_rom_trace(
        tmp_path,
        copy_observations,
        num_threads,
        123,
        actions,
    )

    _assert_native_traces_equal(first, second)


def _native_render_skip_trace(tmp_path, *, disable_render_skip, maxpool_last_two):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    if disable_render_skip:
        os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = "1"
    else:
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    done_info = _done_on_frame_info_path(tmp_path)
    env = StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        4,
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(done_info),
        scenario=str(done_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=maxpool_last_two,
        num_threads=2,
        copy_observations=True,
        info_mode="all",
    )
    try:
        env.seed(20260616)
        obs = env.reset()
        trace = [(_sha(obs), None, None, None)]
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        terminal_count = 0
        for _ in range(12):
            obs, rewards, dones, infos = env.step(actions)
            terminal_count += sum("terminal_observation" in info for info in infos)
            trace.append(
                (
                    _sha(obs),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos(infos),
                ),
            )
        assert terminal_count > 0
        return trace
    finally:
        env.close()
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)


@pytest.mark.parametrize("maxpool_last_two", [True, False])
def test_stable_retro_native_vec_env_nes_render_skip_matches_full_render(
    tmp_path,
    maxpool_last_two,
):
    pytest.importorskip("stable_baselines3")

    baseline = _native_render_skip_trace(
        tmp_path,
        disable_render_skip=True,
        maxpool_last_two=maxpool_last_two,
    )
    render_skip = _native_render_skip_trace(
        tmp_path,
        disable_render_skip=False,
        maxpool_last_two=maxpool_last_two,
    )

    _assert_native_traces_equal(baseline, render_skip)


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


def _mario_native_trace(copy_observations, num_threads, seed, actions, **env_kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    try:
        rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    env = StableRetroNativeVecEnv(
        "SuperMarioBros-Nes-v0",
        8,
        state="Level1-1",
        rom_path=rom_path,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=num_threads,
        copy_observations=copy_observations,
        info_mode="all",
        **env_kwargs,
    )
    try:
        env.seed(seed)
        obs = env.reset()
        trace = [(_sha(obs), None, None, None)]
        for action in actions:
            obs, rewards, dones, infos = env.step(action)
            trace.append(
                (
                    _sha(obs),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos(infos),
                ),
            )
        return trace
    finally:
        env.close()


def _normalize_infos(infos):
    normalized = []
    for info in infos:
        normalized_info = {}
        for key, value in info.items():
            if key == "terminal_observation":
                normalized_info["terminal_observation_sha"] = _sha(value)
                continue
            if key == "reset_info":
                continue
            if hasattr(value, "item"):
                value = value.item()
            normalized_info[key] = value
        normalized.append(normalized_info)
    return normalized


def _assert_native_traces_equal(left, right):
    assert len(left) == len(right)
    for step, (left_item, right_item) in enumerate(zip(left, right)):
        left_obs, left_rewards, left_dones, left_infos = left_item
        right_obs, right_rewards, right_dones, right_infos = right_item
        assert left_obs == right_obs, step
        if left_rewards is not None:
            np.testing.assert_array_equal(left_rewards, right_rewards, err_msg=str(step))
        if left_dones is not None:
            np.testing.assert_array_equal(left_dones, right_dones, err_msg=str(step))
        assert left_infos == right_infos, step


def _native_traces_equal(left, right):
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        left_obs, left_rewards, left_dones, left_infos = left_item
        right_obs, right_rewards, right_dones, right_infos = right_item
        if left_obs != right_obs:
            return False
        if left_rewards is not None and not np.array_equal(left_rewards, right_rewards):
            return False
        if left_dones is not None and not np.array_equal(left_dones, right_dones):
            return False
        if left_infos != right_infos:
            return False
    return True


@pytest.mark.parametrize("copy_observations", [True, False])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_stable_retro_native_vec_env_mario_seed_trace_determinism(
    copy_observations,
    num_threads,
):
    pytest.importorskip("stable_baselines3")

    actions = np.random.default_rng(999).integers(
        0,
        2,
        size=(160, 8, 9),
        dtype=np.uint8,
    )

    first = _mario_native_trace(
        copy_observations,
        num_threads,
        123,
        actions,
        noop_reset_max=1,
        sticky_action_prob=0.25,
    )
    second = _mario_native_trace(
        copy_observations,
        num_threads,
        123,
        actions,
        noop_reset_max=1,
        sticky_action_prob=0.25,
    )

    _assert_native_traces_equal(first, second)


@pytest.mark.parametrize(
    ("num_threads", "noop_reset_max", "sticky_action_prob"),
    [
        (1, 0, 0.0),
        (4, 0, 0.0),
        (4, 1, 0.0),
        (4, 1, 0.25),
    ],
)
@pytest.mark.parametrize("copy_observations", [True, False])
def test_stable_retro_native_vec_env_mario_seed_matrix(
    copy_observations,
    num_threads,
    noop_reset_max,
    sticky_action_prob,
):
    pytest.importorskip("stable_baselines3")

    actions = np.random.default_rng(12345).integers(
        0,
        2,
        size=(96, 8, 9),
        dtype=np.uint8,
    )

    first = _mario_native_trace(
        copy_observations,
        num_threads,
        123,
        actions,
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )
    second = _mario_native_trace(
        copy_observations,
        num_threads,
        123,
        actions,
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )

    _assert_native_traces_equal(first, second)


@pytest.mark.parametrize("copy_observations", [True, False])
def test_stable_retro_native_vec_env_mario_noop_seed_divergence(copy_observations):
    pytest.importorskip("stable_baselines3")

    actions = np.random.default_rng(777).integers(
        0,
        2,
        size=(160, 8, 9),
        dtype=np.uint8,
    )

    seed_123_first = _mario_native_trace(
        copy_observations,
        4,
        123,
        actions,
        noop_reset_max=1,
    )
    seed_123_second = _mario_native_trace(
        copy_observations,
        4,
        123,
        actions,
        noop_reset_max=1,
    )
    seed_456 = _mario_native_trace(
        copy_observations,
        4,
        456,
        actions,
        noop_reset_max=1,
    )

    _assert_native_traces_equal(seed_123_first, seed_123_second)
    assert not _native_traces_equal(seed_123_first, seed_456)

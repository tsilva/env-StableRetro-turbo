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


def _life_counter_info_path(tmp_path):
    life_info = tmp_path / "life_counter_info.json"
    life_info.write_text(
        """
{
  "info": {
    "lives": {
      "address": 0,
      "type": "|u1"
    }
  }
}
""",
        encoding="utf-8",
    )
    return life_info


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


def _make_dr88_native_vec_env(tmp_path, info_path, **kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    num_envs = kwargs.pop("num_envs", 2)
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"

    return StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        num_envs,
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(info_path),
        scenario=str(info_path),
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


def test_stable_retro_native_vec_env_chw_observation_layout(tmp_path):
    pytest.importorskip("stable_baselines3")

    env = _make_test_native_vec_env(
        tmp_path,
        obs_layout="chw",
        info_mode="terminal",
    )
    try:
        obs = env.reset()
        assert obs.shape == (2, 4, 84, 84)
        assert obs.dtype == np.uint8
        assert env.observation_space.shape == (4, 84, 84)
        assert all(env.observation_space.contains(obs[i]) for i in range(env.num_envs))

        actions = np.asarray([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, 4, 84, 84)
        assert rewards.shape == (2,)
        assert rewards.dtype == np.float32
        assert dones.tolist() == [False, False]
        assert len(infos) == 2
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


def test_stable_retro_native_vec_env_life_loss_requires_variable(tmp_path):
    pytest.importorskip("stable_baselines3")

    with pytest.raises(
        ValueError,
        match="life_variable is required when terminate_on_life_loss=True",
    ):
        _make_test_native_vec_env(tmp_path, terminate_on_life_loss=True)

    with pytest.raises(RuntimeError, match="unknown life_variable: missing_lives"):
        _make_test_native_vec_env(
            tmp_path,
            terminate_on_life_loss=True,
            life_variable="missing_lives",
        )


def _mario_rom_path_or_skip():
    import stable_retro as retro

    try:
        return retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")


def _make_mario_native_vec_env(num_envs, rom_path, **kwargs):
    from stable_retro.vec_env import StableRetroNativeVecEnv

    return StableRetroNativeVecEnv(
        "SuperMarioBros-Nes-v0",
        num_envs,
        state="Level1-1",
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=8,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=max(1, num_envs),
        **kwargs,
    )


def test_stable_retro_native_vec_env_life_loss_default_disabled():
    pytest.importorskip("stable_baselines3")

    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_native_vec_env(
        1,
        rom_path,
        info_mode="terminal",
        life_variable="lives",
    )
    enabled_env = _make_mario_native_vec_env(
        1,
        rom_path,
        info_mode="terminal",
        terminate_on_life_loss=True,
        life_variable="lives",
    )
    try:
        env.reset()
        enabled_env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1
        for _ in range(180):
            _, _, dones, infos = env.step(actions)
            _, _, enabled_dones, enabled_infos = enabled_env.step(actions)
            if not bool(enabled_dones[0]):
                assert dones.tolist() == [False]
                continue

            assert enabled_infos[0]["life_loss"] is True
            assert dones.tolist() == [False]
            assert "life_loss" not in infos[0]
            assert "terminal_observation" not in infos[0]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        env.close()
        enabled_env.close()


def test_stable_retro_native_vec_env_life_loss_autoresets_only_one_lane():
    pytest.importorskip("stable_baselines3")

    rom_path = _mario_rom_path_or_skip()
    life_env = _make_mario_native_vec_env(
        2,
        rom_path,
        info_mode="terminal",
        terminate_on_life_loss=True,
        life_variable="lives",
    )
    baseline_env = _make_mario_native_vec_env(
        2,
        rom_path,
        info_mode="terminal",
    )
    try:
        life_env.reset()
        baseline_env.reset()
        actions = np.zeros((life_env.num_envs, life_env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1

        for _ in range(180):
            obs, _, dones, infos = life_env.step(actions)
            baseline_obs, _, baseline_dones, _ = baseline_env.step(actions)
            assert baseline_dones.tolist() == [False, False]
            if not bool(dones[0]):
                assert dones.tolist() == [False, False]
                continue

            assert dones.tolist() == [True, False]
            np.testing.assert_array_equal(obs[1], baseline_obs[1])
            assert infos[0]["life_loss"] is True
            assert infos[0]["died"] is True
            assert infos[0]["life_variable"] == "lives"
            assert infos[0]["current_lives"] < infos[0]["previous_lives"]
            assert "terminal_observation" in infos[0]
            assert "terminal_observation" not in infos[1]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        life_env.close()
        baseline_env.close()


def test_stable_retro_native_vec_env_selected_info_keys(tmp_path):
    pytest.importorskip("stable_baselines3")
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    info_path = _time_reward_info_path(tmp_path)

    common_kwargs = dict(
        state=retro.State.NONE,
        rom_path=str(rom_path),
        info=str(info_path),
        scenario=str(info_path),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=2,
        frame_stack=4,
        num_threads=1,
        info_mode="all",
    )
    default_env = StableRetroNativeVecEnv("Dr88-FamiconIntro", 1, **common_kwargs)
    selected_env = StableRetroNativeVecEnv(
        "Dr88-FamiconIntro",
        1,
        info_keys=["frame_reward_source"],
        **common_kwargs,
    )
    try:
        default_env.reset()
        selected_env.reset()
        action = np.zeros((1, default_env.num_buttons), dtype=np.uint8)
        default_obs, default_rewards, default_dones, default_infos = default_env.step(action)
        selected_obs, selected_rewards, selected_dones, selected_infos = selected_env.step(action)

        np.testing.assert_array_equal(default_obs, selected_obs)
        np.testing.assert_array_equal(default_rewards, selected_rewards)
        np.testing.assert_array_equal(default_dones, selected_dones)
        assert selected_infos == [
            {"frame_reward_source": default_infos[0]["frame_reward_source"]},
        ]
    finally:
        default_env.close()
        selected_env.close()


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


def _as_hwc_observation(obs, obs_layout):
    obs = np.asarray(obs)
    if obs_layout == "chw":
        return np.transpose(obs, (0, 2, 3, 1))
    return obs


def _as_hwc_terminal_observation(obs, obs_layout):
    obs = np.asarray(obs)
    if obs_layout == "chw":
        return np.transpose(obs, (1, 2, 0))
    return obs


def _normalize_infos_as_hwc(infos, obs_layout):
    normalized = []
    for info in infos:
        normalized_info = {}
        for key, value in info.items():
            if key == "terminal_observation":
                normalized_info["terminal_observation_sha"] = _sha(
                    _as_hwc_terminal_observation(value, obs_layout),
                )
                continue
            if key == "reset_info":
                continue
            if hasattr(value, "item"):
                value = value.item()
            normalized_info[key] = value
        normalized.append(normalized_info)
    return normalized


def _native_layout_trace(tmp_path, obs_layout, actions):
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

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
        maxpool_last_two=True,
        num_threads=2,
        copy_observations=False,
        info_mode="all",
        obs_layout=obs_layout,
    )
    try:
        env.seed(20260617)
        obs = env.reset()
        trace = [(_sha(_as_hwc_observation(obs, obs_layout)), None, None, None)]
        terminal_count = 0
        for action in actions:
            obs, rewards, dones, infos = env.step(action)
            terminal_count += sum("terminal_observation" in info for info in infos)
            trace.append(
                (
                    _sha(_as_hwc_observation(obs, obs_layout)),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos_as_hwc(infos, obs_layout),
                ),
            )
        assert terminal_count > 0
        return trace
    finally:
        env.close()


def test_stable_retro_native_vec_env_chw_matches_hwc_trace(tmp_path):
    pytest.importorskip("stable_baselines3")

    actions = np.random.default_rng(20260617).integers(
        0,
        2,
        size=(12, 4, 9),
        dtype=np.uint8,
    )

    hwc = _native_layout_trace(tmp_path, "hwc", actions)
    chw = _native_layout_trace(tmp_path, "chw", actions)

    _assert_native_traces_equal(hwc, chw)


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


def test_stable_retro_native_vec_env_mario_life_decrease_terminates_if_rom_present():
    pytest.importorskip("stable_baselines3")
    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    try:
        rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    env = StableRetroNativeVecEnv(
        "SuperMarioBros-Nes-v0",
        1,
        state="Level1-1",
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=8,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
        info_mode="terminal",
        terminate_on_life_loss=True,
        life_variable="lives",
    )
    try:
        env.reset()
        action = np.zeros((1, env.num_buttons), dtype=np.uint8)
        # Hold RIGHT into the first enemy on Level1-1 to force a deterministic
        # first-life-loss terminal transition.
        action[0, 7] = 1
        for _ in range(180):
            _, _, dones, infos = env.step(action)
            if not bool(dones[0]):
                continue

            assert infos[0]["life_loss"] is True
            assert infos[0]["died"] is True
            assert infos[0]["life_variable"] == "lives"
            assert infos[0]["current_lives"] < infos[0]["previous_lives"]
            assert "terminal_observation" in infos[0]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        env.close()


def test_stable_retro_native_vec_env_mario_all_info_keys_match_default():
    pytest.importorskip("stable_baselines3")

    mario_keys = [
        "coins",
        "levelHi",
        "levelLo",
        "lives",
        "score",
        "scrolling",
        "time",
        "xscrollHi",
        "xscrollLo",
    ]
    actions = np.random.default_rng(20260618).integers(
        0,
        2,
        size=(48, 8, 9),
        dtype=np.uint8,
    )

    baseline = _mario_native_trace(True, 4, 321, actions)
    selected = _mario_native_trace(True, 4, 321, actions, info_keys=mario_keys)

    _assert_native_traces_equal(baseline, selected)


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

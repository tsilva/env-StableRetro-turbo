import gzip
import hashlib
import inspect
import os
import subprocess
import sys
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


def _single_info(vector_infos, env_num):
    info = {}
    for key, value in vector_infos.items():
        if key.startswith("_"):
            continue
        mask = vector_infos.get(f"_{key}")
        if mask is not None and not bool(mask[env_num]):
            continue
        if isinstance(value, dict):
            info[key] = _single_info(value, env_num)
        else:
            info[key] = value[env_num]
    return info


def _infos_to_list(vector_infos, num_envs):
    return [_single_info(vector_infos, env_num) for env_num in range(num_envs)]


def _step(env, actions):
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert truncations.shape == (env.num_envs,)
    assert not np.any(truncations)
    return obs, rewards, terminations, _infos_to_list(infos, env.num_envs)


def test_retro_vec_env_binding_is_private():
    from stable_retro import _retro

    assert not hasattr(_retro, "RetroVecEnv")
    assert hasattr(_retro, "_RetroVecEnv")
    for name in (
        "initial_state_policy_names",
        "initial_state_weights",
        "set_initial_states",
        "set_initial_state_weights",
    ):
        assert not hasattr(_retro._RetroVecEnv, name)


def test_retro_vec_env_legacy_aliases_are_removed():
    from stable_retro.vec_env import RetroVecEnv

    params = inspect.signature(RetroVecEnv.__init__).parameters
    assert params["use_fire_reset"].default is True
    assert not any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    for name in (
        "copy_observations",
        "unsafe_zero_copy",
        "info_mode",
        "info_keys",
        "done_on_info",
        "done_on",
        "autoreset_mode",
    ):
        assert name not in params


def test_retro_vec_env_public_export():
    import gymnasium as gym
    import stable_retro as retro
    from gymnasium.vector import AutoresetMode

    assert issubclass(retro.RetroVecEnv, gym.vector.VectorEnv)
    assert retro.RetroVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    assert "RetroVectorEnv" not in retro.__all__
    assert not hasattr(retro, "RetroVectorEnv")


def test_retro_vec_env_imports_without_sb3():
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'stable_baselines3' or name.startswith('stable_baselines3.'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import stable_retro
from stable_retro.vec_env import RetroVecEnv
assert stable_retro.RetroVecEnv is RetroVecEnv
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )


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


def _make_test_retro_vec_env(tmp_path, **kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    num_envs = kwargs.pop("num_envs", 2)
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    empty_info = _empty_info_path(tmp_path)

    return RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=num_envs,
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


def test_retro_vec_env_gymnasium_contract(tmp_path):
    from gymnasium.vector import AutoresetMode

    env = _make_test_retro_vec_env(tmp_path, info_filter="terminal")
    try:
        obs, infos = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert isinstance(infos, dict)
        assert env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        assert env.num_envs == 2
        assert env.action_space.shape == (2, *env.single_action_space.shape)
        assert env.observation_space.shape == (
            2,
            *env.single_observation_space.shape,
        )

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        step_result = env.step(actions)
        assert len(step_result) == 5
        obs, rewards, terminations, truncations, infos = step_result
        assert obs.shape == env.observation_space.shape
        assert rewards.shape == (env.num_envs,)
        assert terminations.shape == (env.num_envs,)
        assert truncations.shape == (env.num_envs,)
        assert isinstance(infos, dict)
    finally:
        env.close()


def _make_crop_retro_vec_env(tmp_path, **kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    num_envs = kwargs.pop("num_envs", 1)
    info_path = kwargs.pop("info_path", _empty_info_path(tmp_path))
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    defaults = {
        "obs_resize": None,
        "obs_grayscale": False,
        "frame_skip": 1,
        "frame_stack": 1,
        "num_threads": 1,
        "info_filter": "terminal",
    }
    defaults.update(kwargs)

    return RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=num_envs,
        rom_path=str(rom_path),
        info=str(info_path),
        scenario=str(info_path),
        **defaults,
    )


def test_retro_vec_env_crop_mask_signature_defaults():
    from stable_retro.retro_env import RetroEnv
    from stable_retro.vec_env import RetroVecEnv

    vec_sig = inspect.signature(RetroVecEnv)
    env_sig = inspect.signature(RetroEnv)

    assert vec_sig.parameters["obs_crop_mode"].default == "remove"
    assert vec_sig.parameters["obs_crop_fill"].default == 0
    assert "obs_crop_mode" not in env_sig.parameters
    assert "obs_crop_fill" not in env_sig.parameters


def test_retro_vec_env_crop_mask_preserves_full_canvas_shape(tmp_path):
    full_env = _make_crop_retro_vec_env(tmp_path)
    mask_env = _make_crop_retro_vec_env(
        tmp_path,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
    )
    try:
        full_obs = full_env.reset()[0]
        mask_obs = mask_env.reset()[0]

        assert mask_env.observation_space.shape == full_env.observation_space.shape
        assert mask_obs.shape == full_obs.shape
    finally:
        full_env.close()
        mask_env.close()


def test_retro_vec_env_crop_remove_matches_default(tmp_path):
    default_env = _make_crop_retro_vec_env(
        tmp_path,
        obs_crop=(32, 0, 0, 0),
    )
    explicit_env = _make_crop_retro_vec_env(
        tmp_path,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="remove",
    )
    full_env = _make_crop_retro_vec_env(tmp_path)
    try:
        default_obs = default_env.reset()[0]
        explicit_obs = explicit_env.reset()[0]
        full_obs = full_env.reset()[0]

        np.testing.assert_array_equal(explicit_obs, default_obs)
        assert explicit_obs.shape[1] == full_obs.shape[1] - 32
        assert explicit_env.observation_space.shape == default_env.observation_space.shape
    finally:
        default_env.close()
        explicit_env.close()
        full_env.close()


def test_retro_vec_env_crop_none_ignores_mask_mode(tmp_path):
    default_env = _make_crop_retro_vec_env(tmp_path)
    mask_mode_env = _make_crop_retro_vec_env(
        tmp_path,
        obs_crop=None,
        obs_crop_mode="mask",
        obs_crop_fill=123,
    )
    try:
        default_obs = default_env.reset()[0]
        mask_mode_obs = mask_mode_env.reset()[0]

        np.testing.assert_array_equal(mask_mode_obs, default_obs)
        assert mask_mode_env.observation_space.shape == default_env.observation_space.shape
    finally:
        default_env.close()
        mask_mode_env.close()


def test_retro_vec_env_crop_mask_fills_region_before_postprocess(
    tmp_path,
):
    full_env = _make_crop_retro_vec_env(tmp_path)
    mask_env = _make_crop_retro_vec_env(
        tmp_path,
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=17,
    )
    try:
        full_obs = full_env.reset()[0][0]
        mask_obs = mask_env.reset()[0][0]

        assert mask_obs.shape == full_obs.shape
        assert not np.array_equal(mask_obs, full_obs)
        np.testing.assert_array_equal(mask_obs[:32, :, :], 17)
        np.testing.assert_array_equal(mask_obs[32:, :, :], full_obs[32:, :, :])
    finally:
        full_env.close()
        mask_env.close()


@pytest.mark.parametrize("bad_mode", ["hide", "", "MASKED"])
def test_retro_vec_env_rejects_invalid_crop_mode(tmp_path, bad_mode):
    with pytest.raises(ValueError, match="obs_crop_mode must be 'remove' or 'mask'"):
        _make_crop_retro_vec_env(
            tmp_path,
            obs_crop=(32, 0, 0, 0),
            obs_crop_mode=bad_mode,
        )


@pytest.mark.parametrize("bad_fill", [-1, 256])
def test_retro_vec_env_rejects_invalid_crop_fill(tmp_path, bad_fill):
    with pytest.raises(ValueError, match="obs_crop_fill must be between 0 and 255"):
        _make_crop_retro_vec_env(
            tmp_path,
            obs_crop=(32, 0, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=bad_fill,
        )


def test_retro_vec_env_disabled_autoreset_keeps_terminal_lane_until_reset(tmp_path):
    from gymnasium.vector import AutoresetMode

    done_info = _done_on_frame_info_path(tmp_path)
    manual_env = _make_crop_retro_vec_env(
        tmp_path,
        info_path=done_info,
        obs_resize=(16, 16),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=2,
        info_filter="terminal",
    )
    try:
        manual_obs, manual_reset_infos = manual_env.reset(seed=[123])
        assert manual_env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        assert "frame_reward_source" in _single_info(manual_reset_infos, 0)

        actions = np.zeros((1, manual_env.num_buttons), dtype=np.uint8)
        for _ in range(12):
            manual_result = manual_env.step(actions)
            manual_obs, _, manual_done, _, manual_infos = manual_result
            if bool(manual_done[0]):
                break
        else:
            pytest.fail("Dr88 fixture did not reach the terminal frame")

        manual_info = _single_info(manual_infos, 0)
        assert "final_obs" not in manual_info
        assert "final_info" not in manual_info
        terminal_obs = manual_obs.copy()

        with pytest.raises(RuntimeError, match="pending reset"):
            manual_env.step(actions)

        reset_obs, reset_infos = manual_env.reset(
            options={"reset_mask": np.array([True], dtype=np.bool_)},
        )
        assert reset_obs.shape == terminal_obs.shape
        assert "frame_reward_source" in _single_info(reset_infos, 0)
        manual_env.step(actions)
    finally:
        manual_env.close()


@pytest.mark.parametrize("removed_option", ["done_on", "autoreset_mode"])
def test_retro_vec_env_rejects_removed_constructor_options(tmp_path, removed_option):
    with pytest.raises(TypeError, match=removed_option):
        _make_test_retro_vec_env(tmp_path, **{removed_option: None})


def test_retro_vec_env_masked_reset_preserves_unselected_lane(tmp_path):
    from gymnasium.vector import AutoresetMode

    kwargs = {
        "info_filter": "all",
        "noop_reset_max": 3,
        "sticky_action_prob": 0.5,
    }
    info_path = _time_reward_info_path(tmp_path)
    env = _make_dr88_retro_vec_env(tmp_path, info_path, **kwargs)
    baseline_env = _make_dr88_retro_vec_env(tmp_path, info_path, **kwargs)
    try:
        obs, _ = env.reset(seed=[101, 202])
        baseline_obs, _ = baseline_env.reset(seed=[101, 202])
        np.testing.assert_array_equal(obs, baseline_obs)

        actions = np.zeros((2, env.num_buttons), dtype=np.uint8)
        actions[0, 0] = 1
        actions[1, -1] = 1
        for _ in range(2):
            obs, rewards, terminated, truncated, infos = env.step(actions)
            baseline_result = baseline_env.step(actions)
            np.testing.assert_array_equal(obs, baseline_result[0])
            np.testing.assert_array_equal(rewards, baseline_result[1])
            np.testing.assert_array_equal(terminated, baseline_result[2])
            np.testing.assert_array_equal(truncated, baseline_result[3])

        lane_one_before_reset = obs[1].copy()
        reset_obs, reset_infos = env.reset(
            seed=[303, 999],
            options={"reset_mask": np.array([True, False], dtype=np.bool_)},
        )
        np.testing.assert_array_equal(reset_obs[1], lane_one_before_reset)
        assert "frame_reward_source" in _single_info(reset_infos, 0)
        assert _single_info(reset_infos, 1) == {}

        for _ in range(3):
            result = env.step(actions)
            baseline_result = baseline_env.step(actions)
            np.testing.assert_array_equal(result[0][1], baseline_result[0][1])
            assert result[1][1] == baseline_result[1][1]
            assert result[2][1] == baseline_result[2][1]
            assert result[3][1] == baseline_result[3][1]
            assert _single_info(result[4], 1) == _single_info(baseline_result[4], 1)
    finally:
        env.close()
        baseline_env.close()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"reset_mask": [True, False]}, TypeError, "NumPy array"),
        (
            {"reset_mask": np.array([True], dtype=np.bool_)},
            ValueError,
            "shape",
        ),
        (
            {"reset_mask": np.array([1, 0], dtype=np.uint8)},
            TypeError,
            "dtype",
        ),
        (
            {"reset_mask": np.array([False, False], dtype=np.bool_)},
            ValueError,
            "at least one lane",
        ),
    ],
)
def test_retro_vec_env_reset_mask_validation(tmp_path, options, error, message):
    env = _make_test_retro_vec_env(tmp_path)
    try:
        with pytest.raises(error, match=message):
            env.reset(options=options)
    finally:
        env.close()


def test_retro_vec_env_explicit_start_indices_and_per_lane_seeds():
    from gymnasium.vector import AutoresetMode

    rom_path = _mario_rom_path_or_skip()
    kwargs = {
        "state": {"Level1-1": 0.5, "Level1-2": 0.5},
        "info_filter": "all",
        "noop_reset_max": 3,
        "sticky_action_prob": 0.5,
    }
    env = _make_mario_retro_vec_env(2, rom_path, **kwargs)
    twin = _make_mario_retro_vec_env(2, rom_path, **kwargs)
    mask = np.array([True, False], dtype=np.bool_)
    try:
        obs, _ = env.reset(seed=123)
        twin_obs, _ = twin.reset(seed=[123, 124])
        np.testing.assert_array_equal(obs, twin_obs)

        lane_one_before = obs[1].copy()
        reset_obs, reset_infos = env.reset(
            seed=[777, 999],
            options={
                "reset_mask": mask,
                "start_indices": np.array([1, 999], dtype=np.int32),
            },
        )
        assert env.active_states()[0] == "Level1-2"
        np.testing.assert_array_equal(reset_obs[1], lane_one_before)
        assert _single_info(reset_infos, 0)["start_state"] == "Level1-2"
        assert _single_info(reset_infos, 1) == {}

        sampled_obs, _ = env.reset(
            seed=[888, None],
            options={
                "reset_mask": mask,
                "start_indices": np.array([-1, -1], dtype=np.int32),
            },
        )
        assert env.active_states()[0] in {"Level1-1", "Level1-2"}
        np.testing.assert_array_equal(sampled_obs[1], lane_one_before)
    finally:
        env.close()
        twin.close()


def test_retro_vec_env_manual_reset_validates_before_mutation(tmp_path):
    env = _make_test_retro_vec_env(tmp_path)
    try:
        obs, _ = env.reset(seed=[11, 22])
        before = obs.copy()
        invalid_mask = np.array([True, False], dtype=np.bool_)

        with pytest.raises(ValueError, match="seed sequence length"):
            env.reset(seed=[123], options={"reset_mask": invalid_mask})
        with pytest.raises(TypeError, match="start_indices.*dtype"):
            env.reset(
                options={
                    "reset_mask": invalid_mask,
                    "start_indices": np.array([0, -1], dtype=np.int64),
                },
            )
        with pytest.raises(RuntimeError, match="valid initial-state catalog ind"):
            env.reset(
                options={
                    "reset_mask": invalid_mask,
                    "start_indices": np.array([0, -1], dtype=np.int32),
                },
            )
        with pytest.raises(ValueError, match="unsupported reset option"):
            env.reset(options={"reset_mask": invalid_mask, "unknown": True})

        np.testing.assert_array_equal(env._observations, before)
    finally:
        env.close()


@pytest.mark.parametrize("sticky_action_prob", [-0.01, 1.01])
def test_retro_vec_env_rejects_invalid_sticky_action_prob(
    tmp_path,
    sticky_action_prob,
):
    with pytest.raises((RuntimeError, ValueError), match="sticky_action_prob"):
        _make_test_retro_vec_env(
            tmp_path,
            sticky_action_prob=sticky_action_prob,
        )


def _make_dr88_retro_vec_env(tmp_path, info_path, **kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    num_envs = kwargs.pop("num_envs", 2)
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"

    return RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=num_envs,
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


def test_retro_vec_env_same_process(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        num_envs=4,
        info_filter="terminal",
    )
    try:
        obs = env.reset()[0]
        assert obs.shape == (4, 84, 84, 4)
        assert obs.dtype == np.uint8
        assert all(env.single_observation_space.contains(obs[i]) for i in range(env.num_envs))

        actions = np.asarray([env.single_action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = _step(env, actions)
        assert obs.shape == (4, 84, 84, 4)
        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32
        assert dones.tolist() == [False, False, False, False]
        assert len(infos) == 4
        assert all(env.single_observation_space.contains(obs[i]) for i in range(env.num_envs))
    finally:
        env.close()


def test_retro_vec_env_rgb_array_render_returns_raw_screen(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        num_envs=2,
        render_mode="rgb_array",
        info_filter="terminal",
    )
    try:
        obs = env.reset()[0]
        screen = env.render("rgb_array")
        other_screen = env.native.get_screen(1)

        assert screen.dtype == np.uint8
        assert screen.ndim == 3
        assert screen.shape[2] == 3
        assert screen.shape[:2] != obs.shape[1:3]
        assert other_screen.shape == screen.shape
    finally:
        env.close()


def test_retro_vec_env_rgb_array_render_updates_with_indexed_video(
    tmp_path,
):
    os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = "1"
    env = _make_test_retro_vec_env(
        tmp_path,
        num_envs=1,
        render_mode="rgb_array",
        obs_resize_algorithm="area",
        maxpool_last_two=True,
        info_filter="terminal",
    )
    try:
        env.reset()
        action = np.zeros((1, env.num_buttons), dtype=np.uint8)
        frame_hashes = []
        for _ in range(30):
            frame_hashes.append(_sha(env.render("rgb_array")))
            _step(env, action)

        assert len(set(frame_hashes)) > 1
    finally:
        env.close()
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)


def test_stable_retro_simple_image_viewer_close_ignores_cocoa_shutdown_error():
    from stable_retro.rendering import SimpleImageViewer

    class BrokenCloseWindow:
        def close(self):
            raise AttributeError(
                "'CocoaAlternateEventLoop' object has no attribute "
                "'platform_event_loop'",
            )

    viewer = SimpleImageViewer.__new__(SimpleImageViewer)
    viewer.window = BrokenCloseWindow()
    viewer.isopen = True

    viewer.close()

    assert viewer.isopen is False


def test_retro_vec_env_chw_observation_layout(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        obs_layout="chw",
        info_filter="terminal",
    )
    try:
        obs = env.reset()[0]
        assert obs.shape == (2, 4, 84, 84)
        assert obs.dtype == np.uint8
        assert env.single_observation_space.shape == (4, 84, 84)
        assert env.observation_space.shape == (2, 4, 84, 84)
        assert all(env.single_observation_space.contains(obs[i]) for i in range(env.num_envs))

        actions = np.asarray([env.single_action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = _step(env, actions)
        assert obs.shape == (2, 4, 84, 84)
        assert rewards.shape == (2,)
        assert rewards.dtype == np.float32
        assert dones.tolist() == [False, False]
        assert len(infos) == 2
        assert all(env.single_observation_space.contains(obs[i]) for i in range(env.num_envs))
    finally:
        env.close()


@pytest.mark.parametrize("info_filter", ["terminal", "none"])
def test_retro_vec_env_fast_info_filters(info_filter, tmp_path):
    env = _make_test_retro_vec_env(tmp_path, info_filter=info_filter)
    try:
        env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        _, rewards, dones, infos = _step(env, actions)

        assert rewards.tolist() == [0.0, 0.0]
        assert dones.tolist() == [False, False]
        assert infos == [{}, {}]
    finally:
        env.close()


def test_retro_vec_env_keyword_normalization():
    from stable_retro.vec_env import RetroVecEnv

    assert RetroVecEnv._normalize_obs_copy("copy") == (
        True,
        False,
    )
    assert RetroVecEnv._normalize_obs_copy("safe_view") == (
        False,
        False,
    )
    assert RetroVecEnv._normalize_obs_copy("unsafe_view") == (
        False,
        True,
    )
    assert RetroVecEnv._normalize_info_filter(
        {"mode": "terminal", "keys": ("lives", "x_pos")},
    ) == ("terminal", ["lives", "x_pos"])
    with pytest.raises(ValueError, match="info_filter mode"):
        RetroVecEnv._normalize_info_filter("debug")
    with pytest.raises(ValueError, match="unknown info_filter keys"):
        RetroVecEnv._normalize_info_filter({"mode": "terminal", "extra": True})
    assert RetroVecEnv._normalize_state_sampling_weights(
        {"Level1-1": 1.0, "Level1-2": 3.0},
        ("Level1-1", "Level1-2"),
    ) == [0.25, 0.75]
    assert RetroVecEnv._normalize_state_sampling_weights(
        [0.0, 4.0],
        ("Level1-1", "Level1-2"),
    ) == [0.0, 1.0]
    with pytest.raises(ValueError, match="missing state sampling weights"):
        RetroVecEnv._normalize_state_sampling_weights(
            {"Level1-1": 1.0},
            ("Level1-1", "Level1-2"),
        )
    with pytest.raises(ValueError, match="positive number"):
        RetroVecEnv._normalize_state_sampling_weights(
            {"Level1-1": 0.0, "Level1-2": 0.0},
            ("Level1-1", "Level1-2"),
        )


def test_retro_vec_env_accepts_retro_env_keyword_shape(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        obs_layout="chw",
        obs_copy="safe_view",
        maxpool_last_two=True,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        info_filter="terminal",
    )
    try:
        obs = env.reset()[0]
        assert env.obs_copy == "safe_view"
        assert not hasattr(env, "copy_observations")
        assert not hasattr(env, "unsafe_zero_copy")
        assert obs.shape == (2, 4, 84, 84)

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        obs, rewards, dones, infos = _step(env, actions)

        assert obs.shape == (2, 4, 84, 84)
        assert rewards.shape == (2,)
        assert dones.tolist() == [False, False]
        assert infos == [{}, {}]
    finally:
        env.close()


@pytest.mark.parametrize(
    "legacy_kwarg",
    [
        {"frame_maxpool": True},
        {"reset_noops": 0},
        {"action_sticky_prob": 0.0},
        {"unexpected_param": True},
    ],
)
def test_retro_vec_env_rejects_legacy_vector_keyword_shape(
    tmp_path,
    legacy_kwarg,
):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _make_test_retro_vec_env(tmp_path, **legacy_kwarg)


@pytest.mark.parametrize(
    ("invalid_kwarg", "match"),
    [
        ({"num_envs": 0}, "num_envs"),
        ({"num_threads": 0}, "num_threads"),
        ({"obs_resize": (8.5, 8)}, "obs_resize height"),
        ({"obs_crop": (0, 0, -1, 0)}, "obs_crop left"),
        ({"obs_crop_fill": 256}, "obs_crop_fill"),
        ({"obs_resize_algorithm": "linear"}, "obs_resize_algorithm"),
        ({"frame_skip": 0}, "frame_skip"),
        ({"frame_stack": 1.5}, "frame_stack"),
        ({"noop_reset_max": -1}, "noop_reset_max"),
        ({"sticky_action_prob": float("nan")}, "sticky_action_prob"),
        ({"reward_clip": (1.0, -1.0)}, "reward_clip"),
        ({"info_filter": "debug"}, "info_filter"),
    ],
)
def test_retro_vec_env_rejects_invalid_keyword_values(
    tmp_path,
    invalid_kwarg,
    match,
):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    empty_info = _empty_info_path(tmp_path)
    constructor_kwargs = {
        "num_envs": 2,
        "rom_path": str(rom_path),
        "info": str(empty_info),
        "scenario": str(empty_info),
        "obs_resize": (84, 84),
        "obs_grayscale": True,
        "frame_skip": 2,
        "frame_stack": 4,
        "num_threads": 2,
    }
    constructor_kwargs.update(invalid_kwarg)

    with pytest.raises(ValueError, match=match):
        RetroVecEnv(
            "Dr88-FamiconIntro",
            state=retro.State.NONE,
            **constructor_kwargs,
        )


def test_retro_vec_env_validates_mixed_state_config():
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    resolve = RetroVecEnv._resolve_state_config
    game = "SuperMarioBros-Nes-v0"

    with pytest.raises(ValueError, match="state sequence length must match num_envs"):
        resolve(retro, game, 4, ["Level1-1", "Level1-2"])

    with pytest.raises(ValueError, match="non-negative finite"):
        resolve(
            retro,
            game,
            4,
            {"Level1-1": 1.0, "Level1-2": -0.1},
        )

    with pytest.raises(ValueError, match="positive number"):
        resolve(
            retro,
            game,
            4,
            {"Level1-1": 0.0, "Level1-2": 0.0},
        )

    with pytest.raises(ValueError, match="unknown state"):
        resolve(
            retro,
            game,
            4,
            {"Level1-1": 0.5, "DefinitelyMissing": 0.5},
        )

    with pytest.raises(ValueError, match="empty state"):
        resolve(
            retro,
            game,
            4,
            {"Level1-1": 0.5, "": 0.5},
        )

    states, labels, probs, state_collection = resolve(
        retro,
        game,
        4,
        {"Level1-1": 1.0, "Level1-2": 3.0},
    )
    assert states == ["Level1-1", "Level1-2"]
    assert labels == ["Level1-1", "Level1-2"]
    assert probs == [0.25, 0.75]
    assert state_collection is True

    states, labels, probs, state_collection = resolve(
        retro,
        game,
        4,
        {"Level1-1": 0.0, "Level1-2": 3.0},
    )
    assert states == ["Level1-1", "Level1-2"]
    assert labels == ["Level1-1", "Level1-2"]
    assert probs == [0.0, 1.0]
    assert state_collection is True

    states, labels, probs, state_collection = resolve(
        retro,
        game,
        2,
        ["Level1-1", "Level1-2"],
    )
    assert states == ["Level1-1", "Level1-2"]
    assert labels == ["Level1-1", "Level1-2"]
    assert probs is None
    assert state_collection is True

    states, labels, probs, state_collection = resolve(
        retro,
        game,
        4,
        ["Level1-1", "Level1-2", "Level1-1", "Level1-2"],
    )
    assert states == ["Level1-1", "Level1-2", "Level1-1", "Level1-2"]
    assert labels == ["Level1-1", "Level1-2", "Level1-1", "Level1-2"]
    assert probs is None
    assert state_collection is True

    states, labels, probs, state_collection = resolve(retro, game, 1, "Level1-1")
    assert states == ["Level1-1"]
    assert labels == ["Level1-1"]
    assert probs is None
    assert state_collection is False

    with pytest.raises(TypeError, match="unexpected keyword argument 'states'"):
        RetroVecEnv(game, num_envs=1, states=["Level1-1"])
    with pytest.raises(TypeError, match="unexpected keyword argument 'state_probs'"):
        RetroVecEnv(game, num_envs=1, state_probs=[1.0])


def _mario_rom_path_or_skip():
    import stable_retro as retro

    try:
        return retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")


def _make_mario_retro_vec_env(num_envs, rom_path, **kwargs):
    from stable_retro.vec_env import RetroVecEnv

    state = kwargs.pop("state", "Level1-1")
    return RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state=state,
        num_envs=num_envs,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=8,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=max(1, num_envs),
        **kwargs,
    )


def test_retro_vec_env_single_state_active_indices_if_rom_present():
    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        4,
        rom_path,
        info_filter="none",
    )
    try:
        indices = env.active_state_indices()
        assert indices.dtype == np.int32
        assert indices.shape == (4,)
        assert not indices.flags.writeable
        assert env.active_state_indices() is indices
        assert env.initial_state_names == ("Level1-1",)

        env.reset()
        np.testing.assert_array_equal(indices, np.zeros(4, dtype=np.int32))
        assert env.active_states() == ("Level1-1",) * 4
        with pytest.raises(ValueError):
            indices[0] = 1
    finally:
        env.close()


def test_retro_vec_env_mixed_states_reset_infos_if_rom_present():
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    states = ["Level1-1", "Level1-2", "Level1-3"]
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state=states,
        num_envs=3,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=3,
        info_filter="terminal",
    )
    try:
        _, reset_infos = env.reset()
        reset_infos = _infos_to_list(reset_infos, env.num_envs)
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.arange(len(states), dtype=np.int32),
        )
        assert env.initial_state_names == tuple(states)
        assert env.active_states() == tuple(states)
        assert [info["start_state"] for info in reset_infos] == states
        assert [info["state"] for info in reset_infos] == states

        _, reset_infos = env.reset()
        reset_infos = _infos_to_list(reset_infos, env.num_envs)
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.arange(len(states), dtype=np.int32),
        )
        assert [info["start_state"] for info in reset_infos] == states
    finally:
        env.close()


def test_retro_vec_env_fixed_duplicate_states_are_canonical_if_rom_present():
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    states = ["Level1-1", "Level1-2", "Level1-1", "Level1-2"]
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state=states,
        num_envs=4,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=4,
        info_filter="terminal",
    )
    try:
        _, reset_infos = env.reset()
        reset_infos = _infos_to_list(reset_infos, env.num_envs)
        assert env.initial_state_names == ("Level1-1", "Level1-2")
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.array([0, 1, 0, 1], dtype=np.int32),
        )
        assert env.active_states() == tuple(states)
        assert [info["start_state"] for info in reset_infos] == states
    finally:
        env.close()


def test_retro_vec_env_weighted_states_sample_on_reset_if_rom_present():
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    states = ["Level1-1", "Level1-2", "Level1-3"]
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state={state: 1.0 for state in states},
        num_envs=12,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=4,
        info_filter="terminal",
    )
    try:
        env.seed(20260620)
        indices = env.active_state_indices()
        assert env.initial_state_names == tuple(states)
        seen = set()
        for _ in range(20):
            _, reset_infos = env.reset()
            reset_infos = _infos_to_list(reset_infos, env.num_envs)
            reset_indices = indices.copy()
            assert reset_indices.dtype == np.int32
            assert np.all((0 <= reset_indices) & (reset_indices < len(states)))
            reset_states = [info["start_state"] for info in reset_infos]
            assert set(reset_states).issubset(states)
            assert tuple(reset_states) == env.active_states()
            assert reset_states == [
                env.initial_state_names[int(index)] for index in reset_indices
            ]
            seen.update(reset_states)
        assert seen == set(states)
    finally:
        env.close()


def test_retro_vec_env_selected_info_keys(tmp_path):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

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
        info_filter="all",
    )
    default_env = RetroVecEnv("Dr88-FamiconIntro", num_envs=1, **common_kwargs)
    selected_kwargs = {
        **common_kwargs,
        "info_filter": {"mode": "all", "keys": ["frame_reward_source"]},
    }
    selected_env = RetroVecEnv(
        "Dr88-FamiconIntro",
        num_envs=1,
        **selected_kwargs,
    )
    try:
        default_env.reset()
        selected_env.reset()
        action = np.zeros((1, default_env.num_buttons), dtype=np.uint8)
        default_obs, default_rewards, default_dones, default_infos = _step(default_env, action)
        selected_obs, selected_rewards, selected_dones, selected_infos = _step(selected_env, action)

        np.testing.assert_array_equal(default_obs, selected_obs)
        np.testing.assert_array_equal(default_rewards, selected_rewards)
        np.testing.assert_array_equal(default_dones, selected_dones)
        assert selected_infos == [
            {"frame_reward_source": default_infos[0]["frame_reward_source"]},
        ]
    finally:
        default_env.close()
        selected_env.close()


def test_retro_vec_env_observations_do_not_alias_previous_step(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        obs_copy="safe_view",
        info_filter="terminal",
    )
    try:
        obs0 = env.reset()[0]
        obs0_hash = _sha(obs0)
        obs0_ptr = _ptr(obs0)

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        obs1, _, _, _ = _step(env, actions)

        assert _sha(obs0) == obs0_hash
        assert _ptr(obs1) != obs0_ptr
        assert not np.shares_memory(obs1, obs0)
    finally:
        env.close()


def test_retro_vec_env_unsafe_view_aliases_observations(tmp_path):
    env = _make_test_retro_vec_env(
        tmp_path,
        obs_copy="unsafe_view",
        info_filter="terminal",
    )
    try:
        obs0 = env.reset()[0]
        obs0_ptr = _ptr(obs0)

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        obs1, _, _, _ = _step(env, actions)

        assert _ptr(obs1) == obs0_ptr
        assert np.shares_memory(obs1, obs0)
    finally:
        env.close()


def test_retro_vec_env_forwards_per_env_reset_seeds():
    from stable_retro.vec_env import RetroVecEnv

    class FakeNative:
        def __init__(self):
            self.reset_calls = []

        def reset(self, seed, reset_mask, start_indices):
            self.reset_calls.append(
                (seed, reset_mask.copy(), start_indices.copy()),
            )
            return np.zeros((3, 2, 2, 1), dtype=np.uint8), [{}, {}, {}]

    env = RetroVecEnv.__new__(RetroVecEnv)
    env.native = FakeNative()
    env.num_envs = 3
    env._copy_obs = True
    env._seeds = [11, None, 37]
    env._options = [{}, {}, {}]

    obs = env.reset()[0]

    assert obs.shape == (3, 2, 2, 1)
    assert len(env.native.reset_calls) == 1
    seeds, reset_mask, start_indices = env.native.reset_calls[0]
    assert seeds == [11, None, 37]
    np.testing.assert_array_equal(reset_mask, np.ones(3, dtype=np.bool_))
    np.testing.assert_array_equal(start_indices, np.full(3, -1, dtype=np.int32))
    assert env._seeds == [None, None, None]


def test_retro_vec_env_frame_stack_order(tmp_path):
    env = _make_test_retro_vec_env(tmp_path, info_filter="terminal")
    try:
        obs = env.reset()[0]
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        saw_new_frame = False

        for _ in range(8):
            prev = obs.copy()
            obs, _, _, _ = _step(env, actions)

            assert np.array_equal(obs[..., :-1], prev[..., 1:])
            saw_new_frame = saw_new_frame or not np.array_equal(
                obs[..., -1],
                prev[..., -1],
            )

        assert saw_new_frame
    finally:
        env.close()


def _native_test_rom_trace(tmp_path, obs_copy, num_threads, seed, actions):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    reward_info = _time_reward_info_path(tmp_path)

    env = RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=8,
        rom_path=str(rom_path),
        info=str(reward_info),
        scenario=str(reward_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=num_threads,
        obs_copy=obs_copy,
        noop_reset_max=1,
        sticky_action_prob=0.25,
        info_filter="all",
    )
    try:
        env.seed(seed)
        obs = env.reset()[0]
        trace = [(_sha(obs), None, None, None)]
        total_reward = 0.0
        for action in actions:
            obs, rewards, dones, infos = _step(env, action)
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


@pytest.mark.parametrize("obs_copy", ["copy", "safe_view"])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_retro_vec_env_seed_determinism_ci_rom(
    tmp_path,
    obs_copy,
    num_threads,
):
    actions = np.random.default_rng(20260616).integers(
        0,
        2,
        size=(128, 8, 9),
        dtype=np.uint8,
    )

    first = _native_test_rom_trace(
        tmp_path,
        obs_copy,
        num_threads,
        123,
        actions,
    )
    second = _native_test_rom_trace(
        tmp_path,
        obs_copy,
        num_threads,
        123,
        actions,
    )

    _assert_native_traces_equal(first, second)


def _native_render_skip_trace(tmp_path, *, disable_render_skip, maxpool_last_two):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    if disable_render_skip:
        os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = "1"
    else:
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    done_info = _done_on_frame_info_path(tmp_path)
    env = RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=4,
        rom_path=str(rom_path),
        info=str(done_info),
        scenario=str(done_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=maxpool_last_two,
        num_threads=2,
        obs_copy="copy",
        info_filter="all",
    )
    try:
        env.seed(20260616)
        obs = env.reset()[0]
        trace = [(_sha(obs), None, None, None)]
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        terminal_count = 0
        for _ in range(12):
            obs, rewards, dones, infos = _step(env, actions)
            terminal_count += int(np.count_nonzero(dones))
            trace.append(
                (
                    _sha(obs),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos(infos),
                ),
            )
            if np.any(dones):
                env.reset(options={"reset_mask": dones})
        assert terminal_count > 0
        return trace
    finally:
        env.close()
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)


@pytest.mark.parametrize("maxpool_last_two", [True, False])
def test_retro_vec_env_nes_render_skip_matches_full_render(
    tmp_path,
    maxpool_last_two,
):
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


def _has_stella_core():
    root = Path(__file__).resolve().parents[2]
    return (root / "stable_retro" / "cores" / "stella.json").exists() and any(
        (root / "stable_retro" / "cores").glob("stella_libretro.*"),
    )


def _breakout_rom_path_or_skip():
    import stable_retro as retro

    try:
        return retro.data.get_original_romfile_path("Breakout-Atari2600-v0")
    except FileNotFoundError:
        pytest.skip("Breakout-Atari2600-v0 ROM is not imported locally")


def _make_breakout_vec_env(**kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    if not _has_stella_core():
        pytest.skip("stella core is not built")

    defaults = {
        "state": "Start",
        "num_envs": 1,
        "rom_path": _breakout_rom_path_or_skip(),
        "obs_grayscale": True,
        "frame_skip": 4,
        "frame_stack": 4,
        "maxpool_last_two": True,
        "num_threads": 1,
        "obs_copy": "copy",
        "info_filter": "none",
    }
    defaults.update(kwargs)
    return RetroVecEnv("Breakout-Atari2600-v0", **defaults)


def test_retro_vec_env_atari_fire_reset_matches_one_manual_fire_step():
    no_fire_env = _make_breakout_vec_env(
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
    )
    fire_env = _make_breakout_vec_env(
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=True,
    )
    try:
        no_fire_env.reset(seed=123)
        fire_observation = fire_env.reset(seed=123)[0]
        fire_action = np.zeros((1, no_fire_env.num_buttons), dtype=np.uint8)
        fire_action[:, 0] = 1
        manual_fire_observation = no_fire_env.step(fire_action)[0]

        np.testing.assert_array_equal(fire_observation, manual_fire_observation)
    finally:
        no_fire_env.close()
        fire_env.close()


def test_retro_vec_env_atari_fire_reset_respects_reset_mask():
    env = _make_breakout_vec_env(
        num_envs=2,
        num_threads=2,
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=True,
    )
    control = _make_breakout_vec_env(
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=True,
    )
    try:
        env.reset(seed=[123, 456])
        noops = np.zeros((2, env.num_buttons), dtype=np.uint8)
        before = env.step(noops)[0].copy()
        expected_reset = control.reset(seed=789)[0][0]
        reset_observation = env.reset(
            seed=[789, 1011],
            options={"reset_mask": np.array([True, False])},
        )[0]

        np.testing.assert_array_equal(reset_observation[1], before[1])
        np.testing.assert_array_equal(reset_observation[0], expected_reset)
    finally:
        env.close()
        control.close()


def _scalar_area_resize_grayscale(source, dst_shape):
    src_height, src_width = source.shape
    dst_height, dst_width = dst_shape
    resized = np.empty(dst_shape, dtype=np.uint8)
    for dst_y in range(dst_height):
        src_y0 = dst_y * src_height // dst_height
        src_y1 = max((dst_y + 1) * src_height // dst_height, src_y0 + 1)
        src_y1 = min(src_y1, src_height)
        for dst_x in range(dst_width):
            src_x0 = dst_x * src_width // dst_width
            src_x1 = max((dst_x + 1) * src_width // dst_width, src_x0 + 1)
            src_x1 = min(src_x1, src_width)
            region = source[src_y0:src_y1, src_x0:src_x1]
            resized[dst_y, dst_x] = region.sum(dtype=np.uint32) // region.size
    return resized


@pytest.mark.parametrize("resize_algorithm", ["nearest", "area"])
@pytest.mark.parametrize("obs_layout", ["hwc", "chw"])
@pytest.mark.parametrize("frame_stack", [1, 4])
def test_retro_vec_env_atari_crop_mask_observations_are_visible(
    resize_algorithm,
    obs_layout,
    frame_stack,
):
    crop_fill = 0
    env = _make_breakout_vec_env(
        obs_resize=(84, 84),
        obs_crop=(17, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=crop_fill,
        obs_resize_algorithm=resize_algorithm,
        obs_layout=obs_layout,
        frame_stack=frame_stack,
    )
    try:
        observations = [env.reset()[0]]
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        for step in range(3):
            actions[:, 0] = int(step == 0)
            observations.append(env.step(actions)[0])

        expected_shape = (
            (1, 84, 84, frame_stack)
            if obs_layout == "hwc"
            else (1, frame_stack, 84, 84)
        )
        for observation in observations:
            assert observation.shape == expected_shape
            assert np.any(observation != crop_fill)
        assert len({observation.tobytes() for observation in observations}) > 1
    finally:
        env.close()


def test_retro_vec_env_atari_crop_mask_area_matches_scalar_reference():
    crop_top = 17
    crop_fill = 0
    common = {
        "obs_layout": "hwc",
        "frame_skip": 4,
        "frame_stack": 4,
        "maxpool_last_two": True,
    }
    source_env = _make_breakout_vec_env(
        obs_resize=None,
        obs_crop=None,
        obs_resize_algorithm="nearest",
        **common,
    )
    masked_env = _make_breakout_vec_env(
        obs_resize=(84, 84),
        obs_crop=(crop_top, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=crop_fill,
        obs_resize_algorithm="area",
        **common,
    )
    unmasked_env = _make_breakout_vec_env(
        obs_resize=(84, 84),
        obs_crop=None,
        obs_resize_algorithm="area",
        **common,
    )
    actions = np.zeros((1, source_env.num_buttons), dtype=np.uint8)
    try:
        source_obs = source_env.reset()[0]
        masked_obs = masked_env.reset()[0]
        unmasked_obs = unmasked_env.reset()[0]
        for step in range(4):
            source_gray = source_obs[0, :, :, 0]
            masked_source = source_gray.copy()
            masked_source[:crop_top, :] = crop_fill
            masked_reference = _scalar_area_resize_grayscale(
                masked_source,
                (84, 84),
            )
            unmasked_reference = _scalar_area_resize_grayscale(
                source_gray,
                (84, 84),
            )
            masked_gray = masked_obs[0, :, :, 0]
            unmasked_gray = unmasked_obs[0, :, :, 0]

            np.testing.assert_array_equal(masked_gray, masked_reference)
            np.testing.assert_array_equal(unmasked_gray, unmasked_reference)
            fully_masked_rows = sum(
                max((dst_y + 1) * source_gray.shape[0] // 84, 1) <= crop_top
                for dst_y in range(84)
            )
            assert fully_masked_rows > 0
            np.testing.assert_array_equal(
                masked_gray[:fully_masked_rows],
                crop_fill,
            )
            np.testing.assert_array_equal(
                masked_gray[fully_masked_rows:],
                unmasked_gray[fully_masked_rows:],
            )
            assert np.any(masked_gray != crop_fill)

            if step == 3:
                break
            actions[:, 0] = int(step == 0)
            source_obs = source_env.step(actions)[0]
            masked_obs = masked_env.step(actions)[0]
            unmasked_obs = unmasked_env.step(actions)[0]
    finally:
        source_env.close()
        masked_env.close()
        unmasked_env.close()


def test_retro_vec_env_atari_crop_mask_area_general_edges_matches_scalar_reference():
    mask_crop = (3, 4, 5, 6)
    crop_fill = 37
    common = {
        "obs_layout": "hwc",
        "frame_skip": 4,
        "frame_stack": 1,
        "maxpool_last_two": True,
    }
    source_env = _make_breakout_vec_env(
        obs_resize=None,
        obs_crop=None,
        obs_resize_algorithm="nearest",
        **common,
    )
    masked_env = _make_breakout_vec_env(
        obs_resize=(84, 84),
        obs_crop=mask_crop,
        obs_crop_mode="mask",
        obs_crop_fill=crop_fill,
        obs_resize_algorithm="area",
        **common,
    )
    actions = np.zeros((1, source_env.num_buttons), dtype=np.uint8)
    try:
        source_obs = source_env.reset(seed=123)[0]
        masked_obs = masked_env.reset(seed=123)[0]
        for step in range(4):
            source_gray = source_obs[0, :, :, 0]
            masked_source = source_gray.copy()
            top, bottom, left, right = mask_crop
            masked_source[:top, :] = crop_fill
            masked_source[-bottom:, :] = crop_fill
            masked_source[:, :left] = crop_fill
            masked_source[:, -right:] = crop_fill
            expected = _scalar_area_resize_grayscale(masked_source, (84, 84))

            np.testing.assert_array_equal(masked_obs[0, :, :, 0], expected)
            assert np.any(expected != crop_fill)

            if step == 3:
                break
            actions[:, 0] = int(step == 0)
            source_obs = source_env.step(actions)[0]
            masked_obs = masked_env.step(actions)[0]
    finally:
        source_env.close()
        masked_env.close()


def test_retro_vec_env_atari_crop_mask_area_indexed_matches_framebuffer(monkeypatch):
    common = {
        "obs_resize": (84, 84),
        "obs_crop": (17, 0, 0, 0),
        "obs_crop_mode": "mask",
        "obs_crop_fill": 0,
        "obs_resize_algorithm": "area",
        "obs_layout": "chw",
        "frame_skip": 4,
        "frame_stack": 4,
        "maxpool_last_two": True,
        "info_filter": "all",
    }

    def trace(*, indexed: bool):
        if indexed:
            monkeypatch.delenv("STABLE_RETRO_DISABLE_ATARI_INDEXED_VIDEO", raising=False)
        else:
            monkeypatch.setenv("STABLE_RETRO_DISABLE_ATARI_INDEXED_VIDEO", "1")
        env = _make_breakout_vec_env(**common)
        actions = np.zeros((1, env.num_buttons), dtype=np.uint8)
        try:
            observations = [env.reset(seed=123)[0].copy()]
            for step in range(8):
                actions[:, 0] = int(step == 0)
                observations.append(env.step(actions)[0].copy())
            return observations
        finally:
            env.close()

    framebuffer_trace = trace(indexed=False)
    indexed_trace = trace(indexed=True)
    for indexed_obs, framebuffer_obs in zip(
        indexed_trace,
        framebuffer_trace,
        strict=True,
    ):
        np.testing.assert_array_equal(indexed_obs, framebuffer_obs)


def test_stella_breakout_autodetects_ntsc_palette():
    if not _has_stella_core():
        pytest.skip("stella core is not built")

    import stable_retro as retro

    emulator = retro.RetroEmulator(_breakout_rom_path_or_skip())
    state_path = retro.data.get_file_path(
        "Breakout-Atari2600-v0",
        "Start.state",
        retro.data.Integrations.STABLE,
    )
    try:
        with gzip.open(state_path, "rb") as state_file:
            emulator.set_state(state_file.read())
        for _ in range(300):
            emulator.step()

        assert emulator.get_screen_rate() == pytest.approx(59.92, abs=0.01)
        colors = {
            tuple(color)
            for color in np.unique(emulator.get_screen().reshape(-1, 3), axis=0)
        }
        assert {(200, 72, 72), (72, 160, 72), (64, 72, 200)} <= colors
    finally:
        del emulator


def test_stella_reports_the_submitted_frame_width():
    if not _has_stella_core():
        pytest.skip("stella core is not built")

    import stable_retro as retro

    root = Path(__file__).resolve().parents[1]
    emulator = retro.RetroEmulator(str(root / "roms" / "automaton.a26"))
    try:
        assert emulator.get_resolution()[0] == 160
        assert emulator.get_screen().shape[1] == 160
    finally:
        del emulator


def _native_atari_render_skip_trace(tmp_path, *, disable_render_skip, state_path):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    if disable_render_skip:
        os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = "1"
    else:
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "automaton.a26"
    empty_info = _empty_info_path(tmp_path)
    env = RetroVecEnv(
        "Breakout-Atari2600-v0",
        state=str(state_path),
        num_envs=2,
        rom_path=str(rom_path),
        info=str(empty_info),
        scenario=str(empty_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=2,
        obs_copy="copy",
        info_filter="all",
    )
    try:
        env.seed(20260709)
        obs = env.reset()[0]
        trace = [(_sha(obs), None, None, None)]
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        for _ in range(12):
            obs, rewards, dones, infos = _step(env, actions)
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
        os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)


def test_retro_vec_env_atari_render_skip_preserves_control_trace(tmp_path):
    if not _has_stella_core():
        pytest.skip("stella core is not built")

    import stable_retro as retro

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "automaton.a26"
    emulator = retro.RetroEmulator(str(rom_path))
    state_path = tmp_path / "Start.state"
    state_path.write_bytes(gzip.compress(emulator.get_state(), mtime=0))
    del emulator

    baseline = _native_atari_render_skip_trace(
        tmp_path,
        disable_render_skip=True,
        state_path=state_path,
    )
    render_skip = _native_atari_render_skip_trace(
        tmp_path,
        disable_render_skip=False,
        state_path=state_path,
    )

    # automaton.a26 intentionally does not render a game frame, so its initial
    # framebuffer is not a meaningful pixel oracle for Stella. Rewards,
    # termination, and info must still be identical across the fast path.
    assert len(baseline) == len(render_skip)
    for left, right in zip(baseline, render_skip):
        _, left_rewards, left_dones, left_infos = left
        _, right_rewards, right_dones, right_infos = right
        if left_rewards is not None:
            np.testing.assert_array_equal(left_rewards, right_rewards)
            np.testing.assert_array_equal(left_dones, right_dones)
            assert left_infos == right_infos


def test_retro_vec_env_atari_repeated_state_resets_are_deterministic(tmp_path):
    if not _has_stella_core():
        pytest.skip("stella core is not built")

    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "automaton.a26"
    emulator = retro.RetroEmulator(str(rom_path))
    state_path = tmp_path / "Start.state"
    state_path.write_bytes(gzip.compress(emulator.get_state(), mtime=0))
    del emulator

    empty_info = _empty_info_path(tmp_path)
    env = RetroVecEnv(
        "Breakout-Atari2600-v0",
        state=str(state_path),
        num_envs=2,
        rom_path=str(rom_path),
        info=str(empty_info),
        scenario=str(empty_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=2,
        obs_copy="copy",
        info_filter="none",
    )
    actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
    reset_mask = np.ones(env.num_envs, dtype=np.bool_)
    baseline = None
    try:
        for _ in range(10):
            obs, _ = env.reset(options={"reset_mask": reset_mask})
            trace = [obs.tobytes()]
            for _ in range(8):
                obs, rewards, terminated, truncated, _ = env.step(actions)
                trace.extend(
                    (
                        obs.tobytes(),
                        rewards.tobytes(),
                        terminated.tobytes(),
                        truncated.tobytes(),
                    ),
                )
            if baseline is None:
                baseline = trace
            else:
                assert trace == baseline
    finally:
        env.close()


def _as_hwc_observation(obs, obs_layout):
    obs = np.asarray(obs)
    if obs_layout == "chw":
        return np.transpose(obs, (0, 2, 3, 1))
    return obs


def _normalize_infos_as_hwc(infos, obs_layout):
    normalized = []
    for info in infos:
        normalized_info = {}
        for key, value in info.items():
            if hasattr(value, "item"):
                value = value.item()
            normalized_info[key] = value
        normalized.append(normalized_info)
    return normalized


def _native_layout_trace(tmp_path, obs_layout, actions):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    root = Path(__file__).resolve().parents[1]
    rom_path = root / "roms" / "Dr88-FamiconIntro.nes"
    done_info = _done_on_frame_info_path(tmp_path)
    env = RetroVecEnv(
        "Dr88-FamiconIntro",
        state=retro.State.NONE,
        num_envs=4,
        rom_path=str(rom_path),
        info=str(done_info),
        scenario=str(done_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=2,
        obs_copy="safe_view",
        info_filter="all",
        obs_layout=obs_layout,
    )
    try:
        env.seed(20260617)
        obs = env.reset()[0]
        trace = [(_sha(_as_hwc_observation(obs, obs_layout)), None, None, None)]
        terminal_count = 0
        for action in actions:
            obs, rewards, dones, infos = _step(env, action)
            terminal_count += int(np.count_nonzero(dones))
            trace.append(
                (
                    _sha(_as_hwc_observation(obs, obs_layout)),
                    rewards.copy(),
                    dones.copy(),
                    _normalize_infos_as_hwc(infos, obs_layout),
                ),
            )
            if np.any(dones):
                env.reset(options={"reset_mask": dones})
        assert terminal_count > 0
        return trace
    finally:
        env.close()


def test_retro_vec_env_chw_matches_hwc_trace(tmp_path):
    actions = np.random.default_rng(20260617).integers(
        0,
        2,
        size=(12, 4, 9),
        dtype=np.uint8,
    )

    hwc = _native_layout_trace(tmp_path, "hwc", actions)
    chw = _native_layout_trace(tmp_path, "chw", actions)

    _assert_native_traces_equal(hwc, chw)


def test_retro_vec_env_mario_infos_if_rom_present():
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

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

    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        num_envs=1,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
    )
    terminal_env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        num_envs=1,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
        info_filter="terminal",
    )
    try:
        env.reset()
        terminal_env.reset()
        action = np.zeros((1, env.num_buttons), dtype=np.uint8)
        _, rewards, dones, infos = _step(env, action)
        _, terminal_rewards, terminal_dones, terminal_infos = _step(terminal_env, action)

        assert rewards.tolist() == terminal_rewards.tolist()
        assert dones.tolist() == terminal_dones.tolist()
        assert terminal_infos == [{}]
        assert len(infos) == 1
        assert expected_keys.issubset(infos[0])
        assert all(isinstance(infos[0][key], (int, np.integer)) for key in expected_keys)
    finally:
        env.close()
        terminal_env.close()


def test_stable_retro_hf_mario_policy_playback_command_parses():
    from stable_retro.scripts.play_hf_policy import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--policy",
            "/tmp/hf-smb-policy/ppo_supermariobros-nes-v0_4500000_steps.zip",
            "--event",
            "level_change",
            "--seed",
            "10007",
            "--max-steps",
            "2500",
            "--episodes",
            "10",
            "--fps",
            "30",
            "--max-width",
            "672",
        ],
    )

    assert str(args.policy) == (
        "/tmp/hf-smb-policy/ppo_supermariobros-nes-v0_4500000_steps.zip"
    )
    assert args.event == "level_change"
    assert args.seed == 10007
    assert args.max_steps == 2500
    assert args.episodes == 10
    assert args.fps == 30
    assert args.max_width == 672
    assert args.deterministic is False


def test_retro_vec_env_mario_all_info_keys_match_default():
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

    baseline = _mario_native_trace("copy", 4, 321, actions)
    selected = _mario_native_trace(
        "copy",
        4,
        321,
        actions,
        info_filter={"mode": "all", "keys": mario_keys},
    )

    _assert_native_traces_equal(baseline, selected)


def _mario_native_trace(obs_copy, num_threads, seed, actions, **env_kwargs):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    try:
        rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    num_envs = int(env_kwargs.pop("num_envs", 8))
    frame_skip = int(env_kwargs.pop("frame_skip", 4))
    maxpool_last_two = bool(env_kwargs.pop("maxpool_last_two", True))
    info_filter = env_kwargs.pop("info_filter", "all")

    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        num_envs=num_envs,
        rom_path=rom_path,
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=frame_skip,
        frame_stack=4,
        maxpool_last_two=maxpool_last_two,
        num_threads=num_threads,
        obs_copy=obs_copy,
        info_filter=info_filter,
        **env_kwargs,
    )
    try:
        env.seed(seed)
        obs = env.reset()[0]
        trace = [(_sha(obs), None, None, None)]
        for action in actions:
            assert action.shape == (num_envs, env.num_buttons)
            obs, rewards, dones, infos = _step(env, action)
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


@pytest.mark.parametrize(
    ("num_envs", "frame_skip"),
    [
        (1, 1),
        (4, 1),
        (4, 4),
    ],
)
def test_retro_vec_env_mario_sticky_extremes_match_effective_actions(
    num_envs,
    frame_skip,
):
    rng = np.random.default_rng(20260619 + num_envs + frame_skip)
    actions = rng.integers(0, 2, size=(40, num_envs, 9), dtype=np.uint8)
    actions[0] = 0
    repeated_first_action = np.repeat(actions[:1], actions.shape[0], axis=0)
    trace_kwargs = dict(
        num_envs=num_envs,
        frame_skip=frame_skip,
        maxpool_last_two=False,
    )

    sticky_trace = _mario_native_trace(
        "copy",
        max(1, num_envs),
        123,
        actions,
        sticky_action_prob=1.0,
        **trace_kwargs,
    )
    expected_trace = _mario_native_trace(
        "copy",
        max(1, num_envs),
        123,
        repeated_first_action,
        sticky_action_prob=0.0,
        **trace_kwargs,
    )
    non_sticky_trace = _mario_native_trace(
        "copy",
        max(1, num_envs),
        123,
        actions,
        sticky_action_prob=0.0,
        **trace_kwargs,
    )

    _assert_native_traces_equal(sticky_trace, expected_trace)
    assert not _native_traces_equal(sticky_trace, non_sticky_trace)


@pytest.mark.parametrize("obs_copy", ["copy", "safe_view"])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_retro_vec_env_mario_seed_trace_determinism(
    obs_copy,
    num_threads,
):
    actions = np.random.default_rng(999).integers(
        0,
        2,
        size=(160, 8, 9),
        dtype=np.uint8,
    )

    first = _mario_native_trace(
        obs_copy,
        num_threads,
        123,
        actions,
        noop_reset_max=1,
        sticky_action_prob=0.25,
    )
    second = _mario_native_trace(
        obs_copy,
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
@pytest.mark.parametrize("obs_copy", ["copy", "safe_view"])
def test_retro_vec_env_mario_seed_matrix(
    obs_copy,
    num_threads,
    noop_reset_max,
    sticky_action_prob,
):
    actions = np.random.default_rng(12345).integers(
        0,
        2,
        size=(96, 8, 9),
        dtype=np.uint8,
    )

    first = _mario_native_trace(
        obs_copy,
        num_threads,
        123,
        actions,
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )
    second = _mario_native_trace(
        obs_copy,
        num_threads,
        123,
        actions,
        noop_reset_max=noop_reset_max,
        sticky_action_prob=sticky_action_prob,
    )

    _assert_native_traces_equal(first, second)


@pytest.mark.parametrize("obs_copy", ["copy", "safe_view"])
def test_retro_vec_env_mario_noop_seed_divergence(obs_copy):
    actions = np.random.default_rng(777).integers(
        0,
        2,
        size=(160, 8, 9),
        dtype=np.uint8,
    )

    seed_123_first = _mario_native_trace(
        obs_copy,
        4,
        123,
        actions,
        noop_reset_max=1,
    )
    seed_123_second = _mario_native_trace(
        obs_copy,
        4,
        123,
        actions,
        noop_reset_max=1,
    )
    seed_456 = _mario_native_trace(
        obs_copy,
        4,
        456,
        actions,
        noop_reset_max=1,
    )

    _assert_native_traces_equal(seed_123_first, seed_123_second)
    assert not _native_traces_equal(seed_123_first, seed_456)


def test_retro_vec_env_masked_reset_preserves_unselected_lane_trajectory():
    from gymnasium.vector import AutoresetMode

    rom_path = _mario_rom_path_or_skip()
    kwargs = dict(
        state={"Level1-1": 1.0, "Level1-2": 1.0},
        info_filter="all",
        noop_reset_max=3,
        sticky_action_prob=0.5,
        obs_copy="safe_view",
    )
    env = _make_mario_retro_vec_env(4, rom_path, **kwargs)
    control = _make_mario_retro_vec_env(4, rom_path, **kwargs)
    actions = np.zeros((4, env.num_buttons), dtype=np.uint8)
    actions[:, 7] = 1
    try:
        first = env.reset(seed=20260710)[0]
        control_first = control.reset(seed=20260710)[0]
        np.testing.assert_array_equal(first, control_first)
        env.step(actions)
        control.step(actions)

        mask = np.array([False, True, False, True], dtype=np.bool_)
        before = env._observations.copy()
        reset_obs, reset_infos = env.reset(
            seed=[999, 1000, 1001, 1002],
            options={"reset_mask": mask},
        )
        np.testing.assert_array_equal(reset_obs[~mask], before[~mask])
        for key in ("x_pos", "state", "start_state"):
            if key in reset_infos:
                assert np.asarray(reset_infos[f"_{key}"])[~mask].tolist() == [False, False]

        env_step = env.step(actions)
        control_step = control.step(actions)
        np.testing.assert_array_equal(env_step[0][~mask], control_step[0][~mask])
        np.testing.assert_array_equal(env_step[1][~mask], control_step[1][~mask])
        np.testing.assert_array_equal(env.active_state_indices()[~mask], control.active_state_indices()[~mask])
    finally:
        env.close()
        control.close()


def test_retro_vec_env_explicit_start_indices_and_validation_are_atomic():
    from gymnasium.vector import AutoresetMode

    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        3,
        rom_path,
        state={"Level1-1": 1.0, "Level1-2": 1.0},
        info_filter="all",
    )
    try:
        obs, _ = env.reset(seed=7)
        before = obs.copy()
        before_states = env.active_states()
        mask = np.array([True, False, True], dtype=np.bool_)
        starts = np.array([1, 999, 0], dtype=np.int32)
        _, infos = env.reset(options={"reset_mask": mask, "start_indices": starts})
        assert env.active_states() == ("Level1-2", before_states[1], "Level1-1")
        assert _single_info(infos, 0)["start_state"] == "Level1-2"
        assert _single_info(infos, 2)["start_state"] == "Level1-1"

        snapshot = env._observations.copy()
        active = env.active_state_indices().copy()
        with pytest.raises((RuntimeError, ValueError), match="start_indices"):
            env.reset(
                options={
                    "reset_mask": np.array([True, False, False], dtype=np.bool_),
                    "start_indices": np.array([9, -1, -1], dtype=np.int32),
                }
            )
        np.testing.assert_array_equal(env._observations, snapshot)
        np.testing.assert_array_equal(env.active_state_indices(), active)

        invalid_masks = [
            [True, False, False],
            np.array([True, False], dtype=np.bool_),
            np.array([1, 0, 0], dtype=np.int8),
            np.zeros(3, dtype=np.bool_),
        ]
        for invalid in invalid_masks:
            with pytest.raises((TypeError, ValueError), match="reset_mask"):
                env.reset(options={"reset_mask": invalid})
        with pytest.raises(ValueError, match="seed sequence"):
            env.reset(seed=[1, 2])
        np.testing.assert_array_equal(before.shape, snapshot.shape)
    finally:
        env.close()

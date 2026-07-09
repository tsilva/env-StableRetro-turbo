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
    assert hasattr(_retro._RetroVecEnv, "set_initial_states")


def test_retro_vec_env_legacy_aliases_are_removed():
    from stable_retro.vec_env import RetroVecEnv

    params = inspect.signature(RetroVecEnv.__init__).parameters
    assert not any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    for name in (
        "copy_observations",
        "unsafe_zero_copy",
        "info_mode",
        "info_keys",
        "done_on_info",
    ):
        assert name not in params


def test_retro_vec_env_public_export():
    import gymnasium as gym
    import stable_retro as retro
    from gymnasium.vector import AutoresetMode

    assert issubclass(retro.RetroVecEnv, gym.vector.VectorEnv)
    assert retro.RetroVecEnv.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
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
        assert env.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
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
    assert env_sig.parameters["obs_crop_mode"].default == "remove"
    assert env_sig.parameters["obs_crop_fill"].default == 0


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


@pytest.mark.parametrize("obs_copy", ["copy", "safe_view", "unsafe_view"])
@pytest.mark.parametrize("obs_layout", ["hwc", "chw"])
@pytest.mark.parametrize("obs_crop_mode", ["remove", "mask"])
def test_retro_vec_env_crop_modes_terminal_layout_semantics(
    tmp_path,
    obs_copy,
    obs_layout,
    obs_crop_mode,
):
    done_info = _done_on_frame_info_path(tmp_path)
    env = _make_crop_retro_vec_env(
        tmp_path,
        info_path=done_info,
        obs_crop=(1, 0, 0, 0),
        obs_crop_mode=obs_crop_mode,
        obs_crop_fill=0,
        obs_resize=(16, 16),
        obs_grayscale=True,
        frame_stack=2,
        obs_copy=obs_copy,
        obs_layout=obs_layout,
        info_filter="all",
    )
    try:
        obs = env.reset()[0]
        assert env.single_observation_space.contains(obs[0])

        actions = np.zeros((1, env.num_buttons), dtype=np.uint8)
        for _ in range(12):
            obs, _rewards, dones, infos = _step(env, actions)
            if bool(dones[0]):
                break
        else:
            pytest.fail("Dr88 fixture did not reach the terminal frame")

        assert dones.tolist() == [True]
        assert env.single_observation_space.contains(obs[0])
        assert "final_obs" in infos[0]
        assert env.single_observation_space.contains(infos[0]["final_obs"])
    finally:
        env.close()


def test_retro_vec_env_same_step_final_obs_and_info(tmp_path):
    done_info = _done_on_frame_info_path(tmp_path)
    env = _make_crop_retro_vec_env(
        tmp_path,
        info_path=done_info,
        obs_resize=(16, 16),
        obs_grayscale=True,
        frame_stack=2,
        info_filter="terminal",
    )
    try:
        obs, reset_infos = env.reset()
        assert obs.shape == env.observation_space.shape
        assert reset_infos == {}

        actions = np.zeros((1, env.num_buttons), dtype=np.uint8)
        for _ in range(12):
            obs, rewards, terminations, truncations, infos = env.step(actions)
            if bool(terminations[0]):
                break
        else:
            pytest.fail("Dr88 fixture did not reach the terminal frame")

        assert not bool(truncations[0])
        assert env.single_observation_space.contains(obs[0])
        assert "final_obs" in infos
        assert "final_info" in infos
        assert bool(infos["_final_obs"][0])
        assert bool(infos["_final_info"][0])
        assert env.single_observation_space.contains(infos["final_obs"][0])
        final_info = _single_info(infos, 0)["final_info"]
        assert "frame_reward_source" in final_info
        assert "terminal_observation" not in infos
        assert "reset_info" not in infos
        assert "TimeLimit.truncated" not in infos
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


def test_retro_vec_env_no_rule_no_done_omits_done_on_info(tmp_path):
    env = _make_test_retro_vec_env(tmp_path, info_filter="all")
    try:
        env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        _, _, dones, infos = _step(env, actions)

        assert dones.tolist() == [False, False]
        assert "done_on_info" not in infos[0]
        assert "done_on_info" not in infos[1]
    finally:
        env.close()


def test_retro_vec_env_done_on_info_validation():
    from stable_retro.vec_env import RetroVecEnv

    def normalize(value):
        return RetroVecEnv._normalize_done_on(value, label="done_on")

    assert normalize(
        {"level_change": [["levelHi", "levelLo"], "change"]},
    ) == (("level_change", "default", ("levelHi", "levelLo"), "change", "reset"),)

    assert normalize(
        {"life_loss": ("health", "decrease")},
    ) == (("life_loss", "default", ("health",), "decrease", "reset"),)

    assert normalize(
        {
            "life_loss": {
                "description": "Player lost a life.",
                "triggers": [
                    {
                        "id": "lives_decrease",
                        "variables": ["lives"],
                        "op": "decrease",
                        "compare": "reset",
                    },
                    {
                        "id": "health_decrease",
                        "variables": ["health"],
                        "op": "decrease",
                    },
                ],
            },
        },
    ) == (
        ("life_loss", "lives_decrease", ("lives",), "decrease", "reset"),
        ("life_loss", "health_decrease", ("health",), "decrease", "reset"),
    )

    with pytest.raises(ValueError, match="done_on ops"):
        normalize(
            {"bad": ("lives", "less")},
        )

    with pytest.raises(ValueError, match="at least one variable"):
        normalize(
            {"bad": ((), "change")},
        )


def test_retro_vec_env_resolves_scenario_done_on_events(tmp_path):
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    scenario_path = retro.data.get_file_path(
        "SuperMarioBros-Nes-v0",
        "scenario.json",
    )
    assert RetroVecEnv._normalize_done_on(
        ["life_loss", "level_change"],
        label="done_on",
        game="SuperMarioBros-Nes-v0",
        scenario_path=scenario_path,
    ) == (
        ("life_loss", "lives_decrease", ("lives",), "decrease", "reset"),
        ("level_change", "level_bytes_changed", ("levelHi", "levelLo"), "change", "reset"),
    )

    assert RetroVecEnv._normalize_done_on(
        {"life_loss": None},
        label="done_on",
        game="SuperMarioBros-Nes-v0",
        scenario_path=scenario_path,
    ) == (("life_loss", "lives_decrease", ("lives",), "decrease", "reset"),)

    smb3_scenario_path = retro.data.get_file_path(
        "SuperMarioBros3-Nes-v0",
        "scenario.json",
    )
    smb3_level_complete_keys = tuple(
        f"levelComplete{offset:02x}" for offset in range(0, 0x40, 4)
    )
    assert RetroVecEnv._normalize_done_on(
        ["life_loss", "level_change"],
        label="done_on",
        game="SuperMarioBros3-Nes-v0",
        scenario_path=smb3_scenario_path,
    ) == (
        ("life_loss", "lives_decrease", ("lives",), "decrease", "reset"),
        (
            "level_change",
            "return_to_map_started",
            ("returnToMap",),
            "increase",
            "reset",
        ),
        (
            "level_change",
            "level_complete_flags_changed",
            smb3_level_complete_keys,
            "change",
            "reset",
        ),
    )

    custom_scenario = tmp_path / "scenario.json"
    custom_scenario.write_text(
        """
{
  "events": {
    "screen_change": {
      "description": "Screen identifier changed.",
      "triggers": [
        {
          "id": "screen_bytes_changed",
          "variables": ["screenHi", "screenLo"],
          "op": "change",
          "compare": "reset"
        },
        {
          "id": "room_changed",
          "variables": "room",
          "op": "change"
        }
      ]
    }
  },
  "done": {
    "variables": {
      "lives": {
        "op": "equal",
        "reference": 0
      }
    }
  },
  "reward": {
    "variables": {
      "score": {
        "reward": 1.0
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    assert RetroVecEnv._normalize_done_on(
        ["screen_change"],
        label="done_on",
        scenario_path=str(custom_scenario),
    ) == (
        ("screen_change", "screen_bytes_changed", ("screenHi", "screenLo"), "change", "reset"),
        ("screen_change", "room_changed", ("room",), "change", "reset"),
    )

    with pytest.raises(ValueError, match="unknown configured event"):
        RetroVecEnv._normalize_done_on(
            ["boss_clear"],
            label="done_on",
            game="SuperMarioBros-Nes-v0",
            scenario_path=scenario_path,
        )


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
    assert RetroVecEnv._normalize_done_on(
        {"life_loss": ("lives", "decrease")},
        label="done_on",
    ) == (("life_loss", "default", ("lives",), "decrease", "reset"),)
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


def test_retro_vec_env_done_on_info_default_disabled():
    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        1,
        rom_path,
        info_filter="terminal",
    )
    enabled_env = _make_mario_retro_vec_env(
        1,
        rom_path,
        info_filter="terminal",
        done_on={"life_loss": ["lives", "decrease"]},
    )
    try:
        env.reset()
        enabled_env.reset()
        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1
        for _ in range(180):
            _, _, dones, infos = _step(env, actions)
            _, _, enabled_dones, enabled_infos = _step(enabled_env, actions)
            if not bool(enabled_dones[0]):
                assert dones.tolist() == [False]
                continue

            assert "life_loss" in enabled_infos[0]["done_on_info"]
            assert dones.tolist() == [False]
            assert "life_loss" not in infos[0]
            assert "done_on_info" not in infos[0]
            assert "final_obs" not in infos[0]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        env.close()
        enabled_env.close()


def test_retro_vec_env_done_on_info_life_loss_autoresets_only_one_lane():
    rom_path = _mario_rom_path_or_skip()
    life_env = _make_mario_retro_vec_env(
        2,
        rom_path,
        info_filter="terminal",
        done_on={"life_loss": ["lives", "decrease"]},
    )
    baseline_env = _make_mario_retro_vec_env(
        2,
        rom_path,
        info_filter="terminal",
    )
    try:
        life_env.reset()
        baseline_env.reset()
        actions = np.zeros((life_env.num_envs, life_env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1

        for _ in range(180):
            obs, _, dones, infos = _step(life_env, actions)
            baseline_obs, _, baseline_dones, _ = _step(baseline_env, actions)
            assert baseline_dones.tolist() == [False, False]
            if not bool(dones[0]):
                assert dones.tolist() == [False, False]
                continue

            assert dones.tolist() == [True, False]
            np.testing.assert_array_equal(obs[1], baseline_obs[1])
            payload = infos[0]["done_on_info"]["life_loss"]
            assert payload["trigger"] == "default"
            assert payload["compare"] == "reset"
            assert payload["op"] == "decrease"
            assert payload["keys"] == ["lives"]
            assert payload["variables"] == ["lives"]
            assert payload["next"][0] < payload["prev"][0]
            assert "final_obs" in infos[0]
            assert "final_obs" not in infos[1]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        life_env.close()
        baseline_env.close()


def test_retro_vec_env_done_on_info_autoresets_only_changed_lane():
    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        2,
        rom_path,
        state=["Level1-1", "Level1-1"],
        info_filter="terminal",
        done_on={"scroll_change": [["xscrollHi", "xscrollLo"], "change"]},
    )
    baseline_env = _make_mario_retro_vec_env(
        2,
        rom_path,
        state=["Level1-1", "Level1-1"],
        info_filter="terminal",
    )
    try:
        env.reset()
        baseline_env.reset()
        actions = np.zeros((2, env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1

        for _ in range(80):
            obs, _, dones, infos = _step(env, actions)
            baseline_obs, _, baseline_dones, _ = _step(baseline_env, actions)
            assert baseline_dones.tolist() == [False, False]
            if not bool(dones[0]):
                assert dones.tolist() == [False, False]
                continue

            assert dones.tolist() == [True, False]
            np.testing.assert_array_equal(obs[1], baseline_obs[1])
            payload = infos[0]["done_on_info"]["scroll_change"]
            assert payload["trigger"] == "default"
            assert payload["compare"] == "reset"
            assert payload["op"] == "change"
            assert payload["keys"] == ["xscrollHi", "xscrollLo"]
            assert payload["variables"] == ["xscrollHi", "xscrollLo"]
            assert len(payload["prev"]) == 2
            assert len(payload["next"]) == 2
            assert payload["prev"] != payload["next"]
            assert "final_obs" in infos[0]
            assert "final_obs" not in infos[1]
            assert "done_on_info" not in infos[1]
            return

        pytest.fail("Mario xscroll did not change within the test step budget")
    finally:
        env.close()
        baseline_env.close()


def test_retro_vec_env_done_on_info_reports_multiple_triggers_same_event():
    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        1,
        rom_path,
        info_filter="terminal",
        done_on={
            "life_loss": {
                "triggers": [
                    {
                        "id": "lives_decrease",
                        "variables": "lives",
                        "op": "decrease",
                    },
                    {
                        "id": "lives_change",
                        "variables": "lives",
                        "op": "change",
                    },
                ],
            },
        },
    )
    try:
        env.reset()
        actions = np.zeros((1, env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1
        for _ in range(180):
            _, _, dones, infos = _step(env, actions)
            if not bool(dones[0]):
                continue

            done_on_info = infos[0]["done_on_info"]
            assert set(done_on_info) == {"life_loss"}
            life_loss = done_on_info["life_loss"]
            assert life_loss["trigger"] == "lives_decrease"
            assert life_loss["op"] == "decrease"
            assert life_loss["compare"] == "reset"
            assert life_loss["variables"] == ["lives"]
            assert life_loss["next"][0] < life_loss["prev"][0]
            assert [trigger["trigger"] for trigger in life_loss["triggers"]] == [
                "lives_decrease",
                "lives_change",
            ]
            assert [trigger["op"] for trigger in life_loss["triggers"]] == [
                "decrease",
                "change",
            ]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        env.close()


def test_retro_vec_env_done_on_info_weighted_state_autoreset_updates_active_index():
    rom_path = _mario_rom_path_or_skip()
    env = _make_mario_retro_vec_env(
        2,
        rom_path,
        state={"Level1-1": 1.0},
        info_filter="terminal",
        done_on={"scroll_change": [["xscrollHi", "xscrollLo"], "change"]},
    )
    try:
        env.seed(20260624)
        env.reset()
        actions = np.zeros((2, env.num_buttons), dtype=np.uint8)
        actions[0, 7] = 1

        for _ in range(80):
            _, _, dones, infos = _step(env, actions)
            if not bool(dones[0]):
                continue

            assert dones.tolist() == [True, False]
            assert env.initial_state_names == ("Level1-1",)
            np.testing.assert_array_equal(
                env.active_state_indices(),
                np.zeros(2, dtype=np.int32),
            )
            assert infos[0]["start_state"] == "Level1-1"
            return

        pytest.fail("Mario xscroll did not change within the test step budget")
    finally:
        env.close()


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


def test_retro_vec_env_set_state_policy_updates_reset_policy_if_rom_present():
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    states = ["Level1-1", "Level1-2", "Level1-3"]
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state={state: 1.0 for state in states},
        num_envs=8,
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
        assert hasattr(env, "set_state_policy")
        assert not hasattr(env, "set_state")
        assert env.initial_state_names == tuple(states)
        env.reset()
        assert set(env.active_states()).issubset(states)

        env.set_state_policy(
            {"Level1-1": 0.0, "Level1-2": 1.0, "Level1-3": 0.0},
        )
        assert env.state_sampling_weights() == {
            "Level1-1": 0.0,
            "Level1-2": 1.0,
            "Level1-3": 0.0,
        }
        assert set(env.active_states()).issubset(states)

        _, reset_infos = env.reset()
        reset_infos = _infos_to_list(reset_infos, env.num_envs)
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.ones(env.num_envs, dtype=np.int32),
        )
        assert env.initial_state_names == tuple(states)
        assert env.active_states() == ("Level1-2",) * env.num_envs
        assert [info["start_state"] for info in reset_infos] == [
            "Level1-2",
        ] * env.num_envs

        env.set_state_policy("Level1-3")
        assert env.active_states() == ("Level1-2",) * env.num_envs
        env.reset()
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.zeros(env.num_envs, dtype=np.int32),
        )
        assert env.initial_state_names == ("Level1-3",)
        assert env.active_states() == ("Level1-3",) * env.num_envs

        fixed_states = ["Level1-1", "Level1-2"] * 4
        env.set_state_policy(fixed_states)
        env.reset()
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.array([0, 1] * 4, dtype=np.int32),
        )
        assert env.initial_state_names == ("Level1-1", "Level1-2")
        assert env.active_states() == tuple(fixed_states)

        env.set_state_policy(["Level1-3"] * env.num_envs)
        assert env.active_states() == tuple(fixed_states)
        env.reset()
        np.testing.assert_array_equal(
            env.active_state_indices(),
            np.zeros(env.num_envs, dtype=np.int32),
        )
        assert env.initial_state_names == ("Level1-3",)
        assert env.active_states() == ("Level1-3",) * env.num_envs
    finally:
        env.close()


def test_retro_vec_env_set_state_policy_applies_on_autoreset_if_rom_present(
    tmp_path,
):
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    done_info = _done_on_frame_info_path(tmp_path)
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state=["Level1-1", "Level1-1"],
        num_envs=2,
        rom_path=rom_path,
        info=str(done_info),
        scenario=str(done_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=2,
        info_filter="terminal",
    )
    try:
        env.reset()
        assert env.active_states() == ("Level1-1", "Level1-1")

        env.set_state_policy({"Level1-2": 1.0})
        assert env.active_states() == ("Level1-1", "Level1-1")

        actions = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        done_lanes = []
        infos = None
        for _ in range(12):
            _, _, dones, infos = _step(env, actions)
            done_lanes = [
                lane
                for lane, done in enumerate(dones)
                if bool(done)
            ]
            if done_lanes:
                break

        assert done_lanes
        active_states = env.active_states()
        for lane in done_lanes:
            assert active_states[lane] == "Level1-2"
            assert env.active_state_indices()[lane] == 0
            assert infos[lane]["start_state"] == "Level1-2"
    finally:
        env.close()


def test_retro_vec_env_active_indices_update_after_autoreset_if_rom_present(
    tmp_path,
):
    from stable_retro.vec_env import RetroVecEnv

    rom_path = _mario_rom_path_or_skip()
    done_info = _done_on_frame_info_path(tmp_path)
    states = ["Level1-1", "Level1-2", "Level1-3"]
    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state={state: 1.0 for state in states},
        num_envs=6,
        rom_path=rom_path,
        info=str(done_info),
        scenario=str(done_info),
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=3,
        info_filter="terminal",
    )
    try:
        env.seed(20260623)
        env.reset()
        action = np.zeros((env.num_envs, env.num_buttons), dtype=np.uint8)
        _, _, dones, infos = _step(env, action)

        assert np.any(dones)
        active_indices = env.active_state_indices().copy()
        assert np.all((0 <= active_indices) & (active_indices < len(states)))
        for lane, done in enumerate(dones):
            if not bool(done):
                continue
            reset_state = infos[lane]["start_state"]
            assert reset_state == env.initial_state_names[int(active_indices[lane])]
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
            self.reset_seeds = []

        def reset(self, seed):
            self.reset_seeds.append(seed)
            return np.zeros((3, 2, 2, 1), dtype=np.uint8), [{}, {}, {}]

    env = RetroVecEnv.__new__(RetroVecEnv)
    env.native = FakeNative()
    env.num_envs = 3
    env._copy_obs = True
    env._seeds = [11, None, 37]
    env._options = [{}, {}, {}]

    obs = env.reset()[0]

    assert obs.shape == (3, 2, 2, 1)
    assert env.native.reset_seeds == [[11, None, 37]]
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
            terminal_count += sum("final_obs" in info for info in infos)
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


def _native_atari_render_skip_trace(tmp_path, *, disable_render_skip):
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
        state=retro.State.NONE,
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


def test_retro_vec_env_atari_render_skip_matches_full_render(tmp_path):
    if not _has_stella_core():
        pytest.skip("stella core is not built")

    baseline = _native_atari_render_skip_trace(
        tmp_path,
        disable_render_skip=True,
    )
    render_skip = _native_atari_render_skip_trace(
        tmp_path,
        disable_render_skip=False,
    )

    _assert_native_traces_equal(baseline, render_skip)


def _as_hwc_observation(obs, obs_layout):
    obs = np.asarray(obs)
    if obs_layout == "chw":
        return np.transpose(obs, (0, 2, 3, 1))
    return obs


def _as_hwc_final_obs(obs, obs_layout):
    obs = np.asarray(obs)
    if obs_layout == "chw":
        return np.transpose(obs, (1, 2, 0))
    return obs


def _normalize_infos_as_hwc(infos, obs_layout):
    normalized = []
    for info in infos:
        normalized_info = {}
        for key, value in info.items():
            if key == "final_obs":
                normalized_info["final_obs_sha"] = _sha(
                    _as_hwc_final_obs(value, obs_layout),
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
            terminal_count += sum("final_obs" in info for info in infos)
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


def test_retro_vec_env_mario_life_decrease_terminates_if_rom_present():
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    try:
        rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    env = RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        num_envs=1,
        rom_path=rom_path,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_skip=8,
        frame_stack=4,
        maxpool_last_two=True,
        num_threads=1,
        info_filter="terminal",
        done_on=["life_loss"],
    )
    try:
        env.reset()
        action = np.zeros((1, env.num_buttons), dtype=np.uint8)
        # Hold RIGHT into the first enemy on Level1-1 to force a deterministic
        # first-life-loss terminal transition.
        action[0, 7] = 1
        for _ in range(180):
            _, _, dones, infos = _step(env, action)
            if not bool(dones[0]):
                continue

            payload = infos[0]["done_on_info"]["life_loss"]
            assert payload["trigger"] == "lives_decrease"
            assert payload["compare"] == "reset"
            assert payload["op"] == "decrease"
            assert payload["keys"] == ["lives"]
            assert payload["variables"] == ["lives"]
            assert payload["next"][0] < payload["prev"][0]
            assert "final_obs" in infos[0]
            return

        pytest.fail("Mario did not lose a life within the test step budget")
    finally:
        env.close()


def test_stable_retro_hf_mario_level1_policy_triggers_level_change_if_available():
    if not hasattr(np, "_core"):
        pytest.skip("HF SB3 checkpoint requires NumPy 2-compatible pickle paths")

    import stable_retro as retro
    from stable_retro.testing.hf_policy import (
        load_sb3_policy,
        make_mario_level1_policy_env,
        resolve_hf_policy_path,
        run_policy_until_event,
    )

    try:
        retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    except FileNotFoundError:
        pytest.skip("SuperMarioBros-Nes-v0 ROM is not imported locally")

    try:
        model_path = resolve_hf_policy_path(
            "tsilva/SuperMarioBros-NES_Level1",
            "ppo_supermariobros-nes-v0_4500000_steps.zip",
            env_var="STABLE_RETRO_HF_POLICY_PATH",
        )
        model = load_sb3_policy(model_path, device="cpu")
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))

    env = make_mario_level1_policy_env(done_on=["level_change"])
    try:
        result = run_policy_until_event(
            model,
            env,
            event_name="level_change",
            episodes=10,
            max_steps=2500,
            seed_start=10007,
            deterministic=False,
        )
    finally:
        env.close()

    assert result is not None
    assert result.episode < 10
    assert result.payload["trigger"] == "level_bytes_changed"
    assert result.payload["compare"] == "reset"
    assert result.payload["op"] == "change"
    assert result.payload["keys"] == ["levelHi", "levelLo"]
    assert result.payload["prev"] == [0, 0]
    assert result.payload["next"] != [0, 0]


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
            if key == "final_obs":
                normalized_info["final_obs_sha"] = _sha(value)
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

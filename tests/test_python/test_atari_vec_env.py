import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_atari_vec_env_public_export():
    import stable_retro as retro
    from stable_retro.atari_vec_env import AtariVecEnv

    assert retro.AtariVecEnv is AtariVecEnv
    assert AtariVecEnv.backend == "atari-v2"


@pytest.mark.parametrize(
    ("game", "expected"),
    [
        ("Breakout-Atari2600-v0", "breakout"),
        ("MsPacMan-Atari2600-v0", "ms_pac_man"),
        ("breakout", "breakout"),
    ],
)
def test_atari_vec_env_game_id(game, expected):
    from stable_retro.atari_vec_env import ale_game_id

    assert ale_game_id(game) == expected


def test_atari_vec_env_rejects_legacy_states():
    import stable_retro as retro

    with pytest.raises(ValueError, match="does not support Stable Retro save states"):
        retro.AtariVecEnv("Breakout-Atari2600-v0", state="Start")
    with pytest.raises(ValueError, match="does not support Stable Retro save states"):
        retro.AtariVecEnv("Breakout-Atari2600-v0", state=retro.State.DEFAULT)


def test_atari_vec_env_matches_direct_alepy_trace():
    from ale_py import AtariVectorEnv, roms
    from gymnasium.vector import AutoresetMode

    import stable_retro as retro

    try:
        roms.get_rom_path("breakout")
    except FileNotFoundError:
        pytest.skip("ale-py Breakout ROM is not installed")

    kwargs = {
        "num_envs": 2,
        "num_threads": 2,
        "repeat_action_probability": 0.0,
        "img_height": 84,
        "img_width": 84,
        "grayscale": True,
        "stack_num": 4,
        "frameskip": 4,
        "maxpool": True,
        "noop_max": 0,
        "episodic_life": False,
        "life_loss_info": False,
        "reward_clipping": False,
        "use_fire_reset": False,
        "autoreset_mode": AutoresetMode.SAME_STEP,
    }
    direct = AtariVectorEnv("breakout", **kwargs)
    wrapped = retro.AtariVecEnv(
        "Breakout-Atari2600-v0",
        state=retro.State.NONE,
        num_envs=2,
        num_threads=2,
        obs_resize=(84, 84),
        obs_grayscale=True,
        frame_stack=4,
        frame_skip=4,
        maxpool_last_two=True,
        noop_reset_max=0,
        episodic_life=False,
        life_loss_info=False,
        reward_clip=False,
        use_fire_reset=False,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    try:
        direct_obs, direct_info = direct.reset(seed=123)
        wrapped_obs, wrapped_info = wrapped.reset(seed=123)
        np.testing.assert_allclose(wrapped_obs, direct_obs, atol=1)
        assert wrapped_info.keys() == direct_info.keys()

        actions = np.array([0, 1], dtype=np.int64)
        for _ in range(8):
            direct_step = direct.step(actions)
            wrapped_step = wrapped.step(actions)
            # The owned backend uses dependency-free integer area resampling;
            # OpenCV can differ by one at rounding boundaries.
            np.testing.assert_allclose(wrapped_step[0], direct_step[0], atol=1)
            for wrapped_value, direct_value in zip(wrapped_step[1:4], direct_step[1:4]):
                np.testing.assert_array_equal(wrapped_value, direct_value)
            assert wrapped_step[4].keys() == direct_step[4].keys()
    finally:
        direct.close()
        wrapped.close()


def _make_tetris(**kwargs):
    import stable_retro as retro

    return retro.AtariVecEnv(
        "tetris",
        noop_reset_max=0,
        use_fire_reset=False,
        reward_clip=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("num_envs", "num_threads"),
    [(1, 1), (16, 1), (16, 4), (7, 3)],
)
def test_atari_vec_env_disabled_construction_threading_matrix(num_envs, num_threads):
    from gymnasium.vector import AutoresetMode

    env = _make_tetris(
        num_envs=num_envs,
        num_threads=num_threads,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    try:
        assert env.autoreset_mode is AutoresetMode.DISABLED
        assert env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        observations, info = env.reset(seed=123)
        assert observations.shape == (num_envs, 4, 84, 84)
        np.testing.assert_array_equal(info["env_id"], np.arange(num_envs))
    finally:
        env.close()


def test_atari_vec_env_disabled_returns_terminal_observation_and_requires_reset():
    from gymnasium.vector import AutoresetMode

    env = _make_tetris(
        num_envs=2,
        num_threads=2,
        max_episode_steps=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    try:
        reset_obs, _ = env.reset(seed=4)
        terminal_obs, _, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.int64),
        )
        assert not terminated.any()
        assert truncated.all()
        assert (info["episode_frame_number"] == 1).all()
        assert np.any(terminal_obs != reset_obs)
        with pytest.raises(RuntimeError, match="reset that lane first"):
            env.step(np.zeros(2, dtype=np.int64))

        mask = np.array([True, False], dtype=np.bool_)
        observations, reset_info = env.reset(options={"reset_mask": mask})
        assert reset_info["episode_frame_number"].tolist() == [0, 1]
        np.testing.assert_array_equal(observations[1], terminal_obs[1])
        with pytest.raises(RuntimeError, match="terminal Atari lane 1"):
            env.step(np.zeros(2, dtype=np.int64))
    finally:
        env.close()


def test_atari_vec_env_noncontiguous_mask_preserves_unselected_trace():
    from gymnasium.vector import AutoresetMode

    kwargs = {
        "num_envs": 4,
        "num_threads": 2,
        "max_episode_steps": 1_000,
        "autoreset_mode": AutoresetMode.DISABLED,
    }
    reset_env = _make_tetris(**kwargs)
    control_env = _make_tetris(**kwargs)
    try:
        reset_env.reset(seed=100)
        control_env.reset(seed=100)
        actions = np.array([0, 1, 2, 3], dtype=np.int64)
        for _ in range(4):
            reset_step = reset_env.step(actions)
            control_step = control_env.step(actions)
            for actual, expected in zip(reset_step[:4], control_step[:4]):
                np.testing.assert_array_equal(actual, expected)

        mask = np.array([False, True, False, True], dtype=np.bool_)
        control_mask = np.zeros(4, dtype=np.bool_)
        reset_obs, reset_info = reset_env.reset(
            seed=np.array([900, 901, 902, 903]),
            options={"reset_mask": mask},
        )
        control_obs, control_info = control_env.reset(
            seed=np.array([1, 2, 3, 4]),
            options={"reset_mask": control_mask},
        )
        np.testing.assert_array_equal(reset_obs[~mask], control_obs[~mask])
        for key in ("lives", "frame_number", "episode_frame_number"):
            np.testing.assert_array_equal(
                reset_info[key][~mask],
                control_info[key][~mask],
            )
        assert (reset_info["episode_frame_number"][mask] == 0).all()

        for _ in range(8):
            reset_step = reset_env.step(actions)
            control_step = control_env.step(actions)
            for actual, expected in zip(reset_step[:4], control_step[:4]):
                np.testing.assert_array_equal(actual[~mask], expected[~mask])
            for key in ("lives", "frame_number", "episode_frame_number"):
                np.testing.assert_array_equal(
                    reset_step[4][key][~mask],
                    control_step[4][key][~mask],
                )
    finally:
        reset_env.close()
        control_env.close()


def test_atari_vec_env_masked_seed_forms_use_lane_indices():
    from gymnasium.vector import AutoresetMode

    first = _make_tetris(num_envs=4, autoreset_mode=AutoresetMode.DISABLED)
    second = _make_tetris(num_envs=4, autoreset_mode=AutoresetMode.DISABLED)
    try:
        first.reset(seed=8)
        second.reset(seed=[8, 9, 10, 11])
        mask = np.array([False, True, False, True], dtype=np.bool_)
        first_obs, _ = first.reset(seed=100, options={"reset_mask": mask})
        second_obs, _ = second.reset(
            seed=[999_999, 101, 777_777, 103],
            options={"reset_mask": mask},
        )
        np.testing.assert_array_equal(first_obs, second_obs)

        selected_seed_obs, _ = first.reset(
            seed=[201, 203],
            options={"reset_mask": mask},
        )
        full_seed_obs, _ = second.reset(
            seed=[1, 201, 2, 203],
            options={"reset_mask": mask},
        )
        np.testing.assert_array_equal(selected_seed_obs, full_seed_obs)

        uint32_seed = int(np.iinfo(np.uint32).max) - 17
        uint32_obs, _ = first.reset(seed=uint32_seed)
        folded_obs, _ = second.reset(seed=uint32_seed & np.iinfo(np.int32).max)
        np.testing.assert_array_equal(uint32_obs, folded_obs)
    finally:
        first.close()
        second.close()


def test_atari_vec_env_masked_reset_deadlock_regression_runs_in_subprocess(tmp_path):
    code = textwrap.dedent(
        """
        import numpy as np
        import stable_retro as retro
        from gymnasium.vector import AutoresetMode

        env = retro.AtariVecEnv(
            "tetris", num_envs=16, num_threads=4, noop_reset_max=0,
            use_fire_reset=False, autoreset_mode=AutoresetMode.DISABLED,
        )
        env.reset(seed=1)
        for lanes in ([0], [1, 4, 9, 15], [2, 6], list(range(16))):
            mask = np.zeros(16, dtype=np.bool_)
            mask[lanes] = True
            observations, info = env.reset(options={"reset_mask": mask})
            assert observations.shape == (16, 4, 84, 84)
            assert info["env_id"].tolist() == list(range(16))
        env.close()
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_atari_vec_env_repeated_asynchronous_disabled_boundaries():
    from gymnasium.vector import AutoresetMode

    env = _make_tetris(
        num_envs=16,
        num_threads=4,
        max_episode_steps=64,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    actions = np.zeros(16, dtype=np.int64)
    try:
        env.reset(seed=500)
        # Proactive lane-local resets establish different frame-counter phases.
        for lane in range(15):
            _, _, terminated, truncated, _ = env.step(actions)
            reset_mask = terminated | truncated
            reset_mask[lane] = True
            env.reset(options={"reset_mask": reset_mask})

        boundaries = 0
        saw_partial_done = False
        while boundaries < 100:
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if done.any():
                saw_partial_done |= not done.all()
                boundaries += int(done.sum())
                env.reset(options={"reset_mask": done})
        assert saw_partial_done
    finally:
        env.close()


@pytest.mark.parametrize(
    "mode",
    ["SameStep", "NextStep"],
)
def test_atari_vec_env_existing_autoreset_modes_remain_compatible(mode):
    from gymnasium.vector import AutoresetMode

    autoreset_mode = AutoresetMode(mode)
    env = _make_tetris(
        num_envs=2,
        max_episode_steps=1,
        autoreset_mode=autoreset_mode,
    )
    try:
        env.reset(seed=2)
        first = env.step(np.zeros(2, dtype=np.int64))
        assert first[3].all()
        if autoreset_mode is AutoresetMode.SAME_STEP:
            assert "final_obs" in first[4]
            assert (first[4]["episode_frame_number"] == 0).all()
        else:
            assert "final_obs" not in first[4]
            second = env.step(np.zeros(2, dtype=np.int64))
            assert not second[2].any()
            assert not second[3].any()
            assert (second[4]["episode_frame_number"] == 0).all()
    finally:
        env.close()

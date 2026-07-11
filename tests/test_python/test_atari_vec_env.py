import numpy as np
import pytest


def test_atari_vec_env_public_export():
    import stable_retro as retro
    from stable_retro.atari_vec_env import AtariVecEnv

    assert retro.AtariVecEnv is AtariVecEnv
    assert AtariVecEnv.backend == "atari-v1"


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


def test_legacy_atari_envs_are_rejected():
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    message = "libretro Atari backend has been removed"
    with pytest.raises(ValueError, match=message):
        retro.make("Breakout-Atari2600-v0")
    with pytest.raises(ValueError, match=message):
        retro.RetroEnv("Breakout-Atari2600-v0")
    with pytest.raises(ValueError, match=message):
        RetroVecEnv("Breakout-Atari2600-v0")


def test_atari_vec_env_matches_direct_alepy_trace():
    import stable_retro as retro
    from ale_py import AtariVectorEnv
    from ale_py import roms
    from gymnasium.vector import AutoresetMode

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
        np.testing.assert_array_equal(wrapped_obs, direct_obs)
        assert wrapped_info.keys() == direct_info.keys()

        actions = np.array([0, 1], dtype=np.int64)
        for _ in range(8):
            direct_step = direct.step(actions)
            wrapped_step = wrapped.step(actions)
            for wrapped_value, direct_value in zip(wrapped_step[:4], direct_step[:4]):
                np.testing.assert_array_equal(wrapped_value, direct_value)
            assert wrapped_step[4].keys() == direct_step[4].keys()
    finally:
        direct.close()
        wrapped.close()

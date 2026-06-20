import os
import pickle

import numpy as np
import pytest

import stable_retro as retro


def supported_test_rom_names():
    rom_dir = os.path.join(os.path.dirname(__file__), "../roms")
    supported_extensions = set(retro.data.EMU_EXTENSIONS)
    return [
        os.path.splitext(rom)[0]
        for rom in os.listdir(rom_dir)
        if os.path.splitext(rom)[1] in supported_extensions
    ]


@pytest.fixture(
    params=supported_test_rom_names(),
)
def generate_test_env(request):
    import stable_retro.data

    path = os.path.join(os.path.dirname(__file__), "../roms")

    get_file_path_fn = stable_retro.data.get_file_path
    get_romfile_path_fn = stable_retro.data.get_romfile_path

    retro.data.get_file_path = lambda game, file, *args, **kwargs: os.path.join(
        path,
        file,
    )
    retro.data.get_romfile_path = lambda game, *args, **kwargs: [
        os.path.join(path, rom) for rom in os.listdir(path) if rom.startswith(game)
    ][0]

    created_env = []

    def create(state=retro.State.NONE, *args, **kwargs):
        kwargs.setdefault("render_mode", "rgb_array")
        env = retro.make(game=request.param, state=state, *args, **kwargs)
        created_env.append(env)  # noqa: F821
        return env

    try:
        yield create
    finally:
        for env in created_env:
            env.close()

        retro.data.get_file_path = get_file_path_fn
        retro.data.get_romfile_path = get_romfile_path_fn


def test_env_create(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")
    assert generate_test_env(info=json_path, scenario=json_path)


@pytest.mark.parametrize("obs_type", [retro.Observations.IMAGE, retro.Observations.RAM])
def test_env_basic(obs_type, generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(info=json_path, scenario=json_path, obs_type=obs_type)

    obs, info = env.reset()
    assert obs in env.observation_space
    assert isinstance(info, dict)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs in env.observation_space

    assert isinstance(rew, float)
    assert rew == 0

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert terminated is False
    assert truncated is False

    assert isinstance(info, dict)


@pytest.mark.parametrize("algorithm", ["nearest", "bilinear", "area", "linear", "box"])
def test_env_image_preprocessing(algorithm, generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(
        info=json_path,
        scenario=json_path,
        render_mode="rgb_array",
        obs_crop=(1, 1, 1, 1),
        obs_resize=(8, 8),
        obs_resize_algorithm=algorithm,
        obs_grayscale=True,
    )

    assert env.observation_space.shape == (8, 8, 1)

    obs, info = env.reset()
    assert obs in env.observation_space
    assert isinstance(info, dict)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs in env.observation_space
    assert isinstance(rew, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_temporal_preprocessing(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(
        info=json_path,
        scenario=json_path,
        render_mode="rgb_array",
        obs_resize=(8, 8),
        obs_grayscale=True,
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        noop_reset_max=2,
        sticky_action_prob=0.25,
        reward_clip=True,
    )

    assert env.observation_space.shape == (8, 8, 4)

    obs, info = env.reset(seed=123)
    assert obs in env.observation_space
    assert isinstance(info, dict)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs in env.observation_space
    assert isinstance(rew, float)
    assert -1.0 <= rew <= 1.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


class _FixedRandom:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        if not self.values:
            raise AssertionError("unexpected random draw")
        return self.values.pop(0)


@pytest.mark.parametrize(
    ("sticky_action_prob", "draws", "expected_second"),
    [
        (0.0, [], [0, 1, 0]),
        (1.0, [0.999], [1, 0, 0]),
        (0.5, [0.0], [1, 0, 0]),
        (0.5, [0.499999], [1, 0, 0]),
        (0.5, [0.5], [0, 1, 0]),
        (0.5, [0.999], [0, 1, 0]),
    ],
)
def test_env_sticky_action_probability_selects_previous_action(
    sticky_action_prob,
    draws,
    expected_second,
):
    env = retro.RetroEnv.__new__(retro.RetroEnv)
    env._last_action = None
    env._sticky_action_prob = sticky_action_prob
    env.np_random = _FixedRandom(draws)

    first = np.array([1, 0, 0], dtype=np.uint8)
    second = np.array([0, 1, 0], dtype=np.uint8)

    np.testing.assert_array_equal(env._select_step_action(first), first)
    np.testing.assert_array_equal(env._select_step_action(second), expected_second)

    if expected_second == [1, 0, 0]:
        np.testing.assert_array_equal(env._last_action, first)
    else:
        np.testing.assert_array_equal(env._last_action, second)


@pytest.mark.parametrize("sticky_action_prob", [-0.01, 1.01])
def test_env_rejects_invalid_sticky_action_prob(
    sticky_action_prob,
    generate_test_env,
):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    with pytest.raises(ValueError, match="sticky_action_prob"):
        generate_test_env(
            info=json_path,
            scenario=json_path,
            sticky_action_prob=sticky_action_prob,
        )


@pytest.mark.parametrize("frame_skip", [1, 4])
def test_env_sticky_action_is_selected_once_before_frame_skip(frame_skip):
    env = retro.RetroEnv.__new__(retro.RetroEnv)
    requested = np.array([0, 1], dtype=np.uint8)
    selected = np.array([1, 0], dtype=np.uint8)
    set_actions = []
    advanced_frames = []

    env.img = np.zeros((1,), dtype=np.uint8)
    env.ram = None
    env._frame_skip = frame_skip
    env._maxpool_last_two = False
    env._obs_type = retro.Observations.IMAGE
    env._reward_clip = False
    env.render_mode = "rgb_array"
    env._select_step_action = lambda action: selected
    env._native_step_repeat_and_process = lambda action: None
    env._set_action = lambda action: set_actions.append(action.copy())

    def advance_one_frame():
        advanced_frames.append(len(advanced_frames))
        return 1.0, False, {"frame": len(advanced_frames)}

    env._advance_one_frame = advance_one_frame
    env._update_obs = lambda: np.array([len(advanced_frames)], dtype=np.uint8)

    obs, rew, terminated, truncated, info = env.step(requested)

    assert len(set_actions) == 1
    np.testing.assert_array_equal(set_actions[0], selected)
    assert advanced_frames == list(range(frame_skip))
    np.testing.assert_array_equal(obs, [frame_skip])
    assert rew == float(frame_skip)
    assert terminated is False
    assert truncated is False
    assert info == {"frame": frame_skip}


def test_env_data(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(info=json_path, scenario=json_path)
    assert isinstance(env.data[env.system], int)

    env.data["foo"] = 1
    assert env.data["foo"] == 1

    env.reset()

    with pytest.raises(KeyError):
        val = env.data["foo"]
        assert val


def test_env_pickle_roundtrip(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(
        info=json_path,
        scenario=json_path,
        render_mode="rgb_array",
    )
    env.reset()

    payload = pickle.dumps(env)
    env.close()

    restored = pickle.loads(payload)
    try:
        obs, info = restored.reset()
        assert obs in restored.observation_space
        assert isinstance(info, dict)

        obs, rew, terminated, truncated, info = restored.step(
            restored.action_space.sample(),
        )
        assert obs in restored.observation_space
        assert isinstance(rew, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        restored.close()


def test_env_pickle_roundtrip_does_not_serialize_open_movie(
    generate_test_env,
    tmp_path,
):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(
        info=json_path,
        scenario=json_path,
        record=str(tmp_path),
        render_mode="rgb_array",
    )
    if env.statename is None:
        env.close()
        pytest.skip("recording path requires a named state in this fixture setup")

    env.reset()
    assert env.movie is not None

    payload = pickle.dumps(env)
    env.close()

    restored = pickle.loads(payload)
    try:
        assert restored.movie is None
        assert restored.movie_path == str(tmp_path)

        restored.reset()
        assert restored.movie is not None
    finally:
        restored.close()

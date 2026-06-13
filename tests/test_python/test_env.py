import os
import pickle

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

from __future__ import annotations

import stable_retro.cli as cli
import stable_retro.examples.interactive as interactive_module
from stable_retro.examples.interactive import RetroInteractive


def test_resolve_platform_prefers_curated_game(monkeypatch):
    monkeypatch.setattr(
        cli,
        "installed_games_by_platform",
        lambda: {"Genesis": ["Airstriker-Genesis-v0", "SonicTheHedgehog-Genesis-v0"]},
    )
    monkeypatch.setattr(cli.retro.data, "list_games", lambda: ["Airstriker-Genesis-v0"])

    assert cli.resolve_target("mega-drive") == [("Genesis", "Airstriker-Genesis-v0")]


def test_resolve_game_id_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(cli, "installed_games_by_platform", lambda: {})
    monkeypatch.setattr(cli.retro.data, "list_games", lambda: ["SuperMarioBros-Nes-v0"])

    assert cli.resolve_target("supermariobros-nes-v0") == [
        ("Nes", "SuperMarioBros-Nes-v0")
    ]


def test_resolve_all_selects_one_game_per_installed_platform(monkeypatch):
    monkeypatch.setattr(
        cli,
        "installed_games_by_platform",
        lambda: {
            "Nes": ["BubbleBobble-Nes-v0", "SuperMarioBros-Nes-v0"],
            "Sms": ["SonicTheHedgehog-Sms-v0"],
        },
    )
    monkeypatch.setattr(cli.retro.data, "list_games", lambda: [])

    assert cli.resolve_target("all") == [
        ("Nes", "SuperMarioBros-Nes-v0"),
        ("Sms", "SonicTheHedgehog-Sms-v0"),
    ]


def test_resolve_snes_uses_verified_smoke_game(monkeypatch):
    monkeypatch.setattr(
        cli,
        "installed_games_by_platform",
        lambda: {
            "Snes": ["SuperMarioWorld-Snes-v0", "TetrisAttack-Snes-v0"],
        },
    )
    monkeypatch.setattr(
        cli.retro.data,
        "list_games",
        lambda: ["SuperMarioWorld-Snes-v0", "TetrisAttack-Snes-v0"],
    )

    assert cli.resolve_target("snes") == [("Snes", "SuperMarioWorld-Snes-v0")]


def test_list_does_not_launch_games(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "installed_games_by_platform",
        lambda: {"Genesis": ["Airstriker-Genesis-v0"]},
    )

    def fail_if_called(*_args):
        raise AssertionError("--list must not launch a game")

    monkeypatch.setattr(cli, "run_game", fail_if_called)

    assert cli.main(["play", "--list"]) == 0
    assert capsys.readouterr().out == "Genesis: Airstriker-Genesis-v0\n"


def test_interactive_sms_pause_button_maps_to_enter():
    interactive = object.__new__(RetroInteractive)
    interactive._buttons = ["B", None, None, "PAUSE", "UP", "DOWN", "LEFT", "RIGHT", "A"]

    assert interactive.keys_to_act(set()) == [False] * 9
    assert interactive.keys_to_act({"ENTER"}) == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_interactive_window_dimensions_preserve_display_aspect_ratio():
    width, height = interactive_module._window_dimensions(
        image_width=256,
        image_height=240,
        aspect_ratio=4 / 3,
        screen_width=1920,
        screen_height=1080,
    )

    assert (width, height) == (1280, 960)
    assert width / height == 4 / 3


def test_environment_aspect_ratio_uses_core_display_metadata():
    class Emulator:
        @staticmethod
        def get_aspect_ratio():
            return 10 / 9

    class Env:
        em = Emulator()

    assert interactive_module._environment_aspect_ratio(Env(), 160, 144) == 10 / 9


def test_retro_interactive_resolves_aspect_ratio_after_reset(monkeypatch):
    captured = {}

    class Emulator:
        @staticmethod
        def get_aspect_ratio():
            return 10 / 9

    class Env:
        buttons = ["A"]
        em = Emulator()

    def capture_init(_self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(interactive_module.retro, "make", lambda **_kwargs: Env())
    monkeypatch.setattr(interactive_module.Interactive, "__init__", capture_init)

    RetroInteractive(game="Game-GameBoy-v0", state="Start", scenario=None, record=False)

    assert captured.get("aspect_ratio") is None
    assert captured.get("show_obs") is False


def test_retro_interactive_show_obs_uses_ppo_preprocessing(monkeypatch):
    env_kwargs = {}
    interactive_kwargs = {}

    class Env:
        buttons = ["A"]

    monkeypatch.setattr(
        interactive_module.retro,
        "make",
        lambda **kwargs: env_kwargs.update(kwargs) or Env(),
    )
    monkeypatch.setattr(
        interactive_module.Interactive,
        "__init__",
        lambda _self, **kwargs: interactive_kwargs.update(kwargs),
    )

    RetroInteractive(
        game="Game-Nes-v0",
        state="Start",
        scenario=None,
        record=False,
        show_obs=True,
    )

    assert {key: env_kwargs[key] for key in (
        "obs_resize",
        "obs_crop",
        "obs_grayscale",
        "obs_resize_algorithm",
    )} == {
        "obs_resize": (84, 84),
        "obs_crop": (32, 0, 0, 0),
        "obs_grayscale": True,
        "obs_resize_algorithm": "area",
    }
    assert interactive_kwargs["tps"] == 60
    assert interactive_kwargs["show_obs"] is True


def test_retro_interactive_samples_observation_stack_without_slower_gameplay():
    interactive = object.__new__(RetroInteractive)
    interactive._observation_sample_interval = 4
    interactive._observation_frame_stack = 4
    interactive._observation_sample_steps = 0
    interactive._observation_frames = []

    initial = interactive_module.np.full((1, 1, 1), 1, dtype=interactive_module.np.uint8)
    image = interactive.get_observation_image(initial, reset=True)
    assert image[0, :, 0].tolist() == [1, 1, 1, 1]

    for value in (2, 3, 4):
        frame = interactive_module.np.full((1, 1, 1), value, dtype=interactive_module.np.uint8)
        assert interactive.get_observation_image(frame) is None

    sampled = interactive_module.np.full((1, 1, 1), 5, dtype=interactive_module.np.uint8)
    image = interactive.get_observation_image(sampled)
    assert image[0, :, 0].tolist() == [1, 1, 1, 5]


def test_cli_player_does_not_record_movies(monkeypatch):
    calls = []

    class Player:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            calls.append("run")

    monkeypatch.setattr(interactive_module, "RetroInteractive", Player)

    cli.run_game("Airstriker-Genesis-v0", cli.retro.State.DEFAULT)

    assert calls == [
        {
            "game": "Airstriker-Genesis-v0",
            "state": cli.retro.State.DEFAULT,
            "scenario": None,
            "record": False,
            "show_obs": False,
        },
        "run",
    ]


def test_cli_show_obs_passes_preprocessed_view_to_player(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "resolve_target", lambda _target: [("Nes", "Game-Nes-v0")])
    monkeypatch.setattr(cli, "run_game", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert cli.main(["play", "nes", "--show-obs"]) == 0
    assert calls == [(("Game-Nes-v0", cli.retro.State.DEFAULT), {"show_obs": True})]


def test_observation_view_tiles_frame_stacked_grayscale_pixels():
    observation = interactive_module.np.array([[[1, 2, 3, 4]]], dtype=interactive_module.np.uint8)

    image = interactive_module._observation_to_rgb(observation)

    assert image.shape == (1, 4, 3)
    assert image[0, :, 0].tolist() == [1, 2, 3, 4]


def test_interactive_close_stops_the_manual_render_loop():
    class Env:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class Window:
        def __init__(self, interactive):
            self.interactive = interactive
            self.switched = False
            self.flipped = False

        def switch_to(self):
            self.switched = True

        def dispatch_events(self):
            self.interactive._on_close()

        def flip(self):
            self.flipped = True

    interactive = object.__new__(RetroInteractive)
    interactive._closed = False
    interactive._env = Env()
    interactive._win = Window(interactive)

    interactive.run()

    assert interactive._env.close_calls == 1
    assert interactive._win.switched is True
    assert interactive._win.flipped is False

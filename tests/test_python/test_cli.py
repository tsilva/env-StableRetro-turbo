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
        },
        "run",
    ]


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

"""Command-line launcher for quickly checking installed Retro platforms."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable

import numpy as np

import env_stableretro_turbo as retro


PREFERRED_GAMES = {
    "Atari2600": "Breakout-Atari2600-v0",
    "GameBoy": "GradiusTheInterstellarAssault-GameBoy-v0",
    "Genesis": "Airstriker-Genesis-v0",
    "Nes": "SuperMarioBros-Nes-v0",
    "Sms": "SonicTheHedgehog-Sms-v0",
    "Snes": "SuperMarioWorld-Snes-v0",
}

PLATFORM_ALIASES = {
    "atari": "Atari2600",
    "atari2600": "Atari2600",
    "gb": "GameBoy",
    "gameboy": "GameBoy",
    "genesis": "Genesis",
    "megadrive": "Genesis",
    "megacd": "SCD",
    "md": "Genesis",
    "n64": "N64",
    "nes": "Nes",
    "nds": "NintendoDS",
    "nintendods": "NintendoDS",
    "nintendo64": "N64",
    "saturn": "Saturn",
    "scd": "SCD",
    "segacd": "SCD",
    "sms": "Sms",
    "mastersystem": "Sms",
    "snes": "Snes",
}

_ATARI_DIFFICULTY_INPUTS = {
    "A": (10, 11),  # Retropad L and R: both console difficulty switches to A.
    "B": (12, 13),  # Retropad L2 and R2: both switches to B.
}


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _parse_state(value: str):
    """Accept save-state names plus friendly spellings of special states."""
    normalized = value.strip().lower()
    if normalized == "default":
        return retro.State.DEFAULT
    if normalized in {"none", "poweron", "power-on"}:
        return retro.State.NONE
    return value


def _parse_startup_press(value: str) -> tuple[str, int]:
    """Parse ``BUTTON`` or ``BUTTON:COUNT`` for repeatable startup inputs."""
    raw_name, separator, raw_count = value.rpartition(":")
    if separator:
        name = raw_name.strip()
        try:
            count = int(raw_count)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"startup button count must be an integer: {value!r}"
            ) from error
    else:
        name = value.strip()
        count = 1
    if not name:
        raise argparse.ArgumentTypeError("startup button name must not be empty")
    if count <= 0:
        raise argparse.ArgumentTypeError("startup button count must be positive")
    return name.upper(), count


def _platform_for_game(game: str) -> str | None:
    game_without_version = re.sub(r"-v\d+$", "", game)
    if "-" not in game_without_version:
        return None
    return game_without_version.rsplit("-", 1)[1]


def installed_games_by_platform() -> dict[str, list[str]]:
    """Return installed games, grouped by their emulator platform."""
    games_by_platform: dict[str, list[str]] = {}
    for game in retro.data.list_games():
        platform = _platform_for_game(game)
        if platform is None:
            continue
        try:
            retro.data.get_romfile_path(game)
        except FileNotFoundError:
            continue
        games_by_platform.setdefault(platform, []).append(game)

    for games in games_by_platform.values():
        games.sort()
    return games_by_platform


def select_game(platform: str, games: Iterable[str]) -> str:
    """Choose a predictable representative game for one platform."""
    available = sorted(games)
    preferred = PREFERRED_GAMES.get(platform)
    if preferred in available:
        return preferred
    return available[0]


def resolve_target(target: str) -> list[tuple[str, str]]:
    """Resolve a platform alias, game ID, or ``all`` into launch targets."""
    installed = installed_games_by_platform()
    all_games = retro.data.list_games()
    games_by_name = {game.lower(): game for game in all_games}
    platforms_by_alias = {
        _normalized(platform): platform
        for platform in (_platform_for_game(game) for game in all_games)
        if platform is not None
    }
    platforms_by_alias.update(PLATFORM_ALIASES)

    if target.lower() == "all":
        if not installed:
            raise ValueError("No imported ROMs were found. Import a ROM before using `all`.")
        return [
            (platform, select_game(platform, installed[platform]))
            for platform in sorted(installed)
        ]

    game = games_by_name.get(target.lower())
    if game is not None:
        platform = _platform_for_game(game) or "unknown platform"
        return [(platform, game)]

    platform = platforms_by_alias.get(_normalized(target))
    if platform is not None:
        games = installed.get(platform, [])
        if games:
            return [(platform, select_game(platform, games))]
        raise ValueError(
            f"No imported ROMs are available for {platform}. "
            "Run `env-stableretro-turbo play --list` to see runnable platforms."
        )

    choices = ", ".join(sorted(platforms_by_alias))
    raise ValueError(
        f"Unknown platform or game ID: {target}. "
        f"Use a game ID, one of: {choices}, or all."
    )


def _button_index(env, requested: str) -> int:
    normalized = requested.upper()
    for index, button in enumerate(env.buttons):
        if button is not None and str(button).upper() == normalized:
            return index
    available = ", ".join(str(button) for button in env.buttons if button is not None)
    raise ValueError(
        f"button {requested!r} is unavailable for {env.gamename}; "
        f"available buttons: {available}"
    )


def _pulse_button(env, requested: str, count: int = 1):
    """Press and release one integration-owned button ``count`` times."""
    index = _button_index(env, requested)
    noop = np.zeros(env.num_buttons, dtype=np.int8)
    pressed = noop.copy()
    pressed[index] = 1
    observation = None
    for _ in range(count):
        env.step(pressed)
        observation = env.step(noop)[0]
    return observation


def _set_atari_difficulty(env, difficulty: str) -> None:
    """Set both Stella difficulty switches through standard Retropad inputs."""
    pressed = np.zeros(16, dtype=np.uint8)
    pressed[list(_ATARI_DIFFICULTY_INPUTS[difficulty])] = 1
    env.em.set_button_mask(pressed, 0)
    env.em.step()
    env.em.set_button_mask(np.zeros(16, dtype=np.uint8), 0)
    env.em.step()


def _apply_startup_configuration(
    env,
    *,
    platform: str,
    mode: int | None,
    difficulty: str | None,
    startup_presses: tuple[tuple[str, int], ...],
):
    """Apply requested console configuration and return the latest observation."""
    observation = None
    if mode is not None or difficulty is not None:
        if platform != "Atari2600":
            raise ValueError(
                "--mode and --difficulty are only supported for Atari2600 games"
            )
        if mode is not None:
            observation = _pulse_button(env, "SELECT", mode)
        if difficulty is not None:
            _set_atari_difficulty(env, difficulty)
        observation = _pulse_button(env, "RESET")

    for button, count in startup_presses:
        observation = _pulse_button(env, button, count)
    return observation


def run_game(
    game: str,
    state,
    *,
    show_obs: bool = False,
    mode: int | None = None,
    difficulty: str | None = None,
    startup_presses: tuple[tuple[str, int], ...] = (),
) -> None:
    """Open the repository's interactive player for a single game."""
    from env_stableretro_turbo.examples.interactive import RetroInteractive

    player = RetroInteractive(
        game=game,
        state=state,
        scenario=None,
        record=False,
        show_obs=show_obs,
        use_restricted_actions=retro.Actions.ALL,
    )
    observation = None
    if mode is not None or difficulty is not None or startup_presses:
        observation = _apply_startup_configuration(
            player._env,
            platform=_platform_for_game(game) or "unknown platform",
            mode=mode,
            difficulty=difficulty,
            startup_presses=startup_presses,
        )
    if observation is not None:
        player._observation = observation
        player._image = player.get_image(observation, player._env)
        if player._observation_window is not None:
            player._observation_window.set_image(
                player.get_observation_image(observation, reset=True)
            )
    player.run()


def _print_runnable_platforms() -> None:
    installed = installed_games_by_platform()
    if not installed:
        print("No imported ROMs found.")
        return
    for platform in sorted(installed):
        print(f"{platform}: {select_game(platform, installed[platform])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="env-stableretro-turbo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    play = subparsers.add_parser("play", help="launch a platform, game ID, or all")
    play.add_argument(
        "target",
        nargs="?",
        help="platform alias, full game ID, or all",
    )
    play.add_argument(
        "--state",
        default=retro.State.DEFAULT,
        type=_parse_state,
        help="save-state name, 'default', or 'none' for the power-on state",
    )
    play.add_argument(
        "--list",
        action="store_true",
        help="show runnable platforms and their selected games",
    )
    play.add_argument(
        "--show-obs",
        action="store_true",
        help="also show the PPO-style preprocessed observation in a second window",
    )
    play.add_argument(
        "--press",
        action="append",
        default=[],
        type=_parse_startup_press,
        metavar="BUTTON[:COUNT]",
        help="press and release a game button before play; may be repeated",
    )
    play.add_argument(
        "--mode",
        type=int,
        metavar="N",
        help="Atari mode value: press SELECT N times, then RESET",
    )
    play.add_argument(
        "--difficulty",
        type=str.upper,
        choices=("A", "B"),
        help="set both Atari difficulty switches before RESET",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "play":
        parser.error(f"Unsupported command: {args.command}")
    if args.list:
        if args.target:
            parser.error("--list does not accept a platform or game ID")
        _print_runnable_platforms()
        return 0
    if not args.target:
        parser.error("play requires a platform, game ID, or all")
    if args.mode is not None and args.mode < 0:
        parser.error("--mode must be non-negative")

    try:
        targets = resolve_target(args.target)
    except ValueError as error:
        parser.error(str(error))

    for platform, game in targets:
        print(f"Launching {platform}: {game}", flush=True)
        try:
            run_game(
                game,
                args.state,
                show_obs=args.show_obs,
                mode=args.mode,
                difficulty=args.difficulty,
                startup_presses=tuple(args.press),
            )
        except (FileNotFoundError, KeyError, ValueError) as error:
            if len(targets) == 1:
                parser.error(str(error))
            print(f"Could not launch {platform}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

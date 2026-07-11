"""Command-line launcher for quickly checking installed Retro platforms."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable

import stable_retro as retro


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


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


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
            "Run `stable-retro-turbo play --list` to see runnable platforms."
        )

    choices = ", ".join(sorted(platforms_by_alias))
    raise ValueError(
        f"Unknown platform or game ID: {target}. "
        f"Use a game ID, one of: {choices}, or all."
    )


def run_game(game: str, state: str, *, show_obs: bool = False) -> None:
    """Open the repository's interactive player for a single game."""
    from stable_retro.examples.interactive import RetroInteractive

    RetroInteractive(
        game=game,
        state=state,
        scenario=None,
        record=False,
        show_obs=show_obs,
    ).run()


def _print_runnable_platforms() -> None:
    installed = installed_games_by_platform()
    if not installed:
        print("No imported ROMs found.")
        return
    for platform in sorted(installed):
        print(f"{platform}: {select_game(platform, installed[platform])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stable-retro-turbo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    play = subparsers.add_parser("play", help="launch a platform, game ID, or all")
    play.add_argument(
        "target",
        nargs="?",
        help="platform alias, full game ID, or all",
    )
    play.add_argument("--state", default=retro.State.DEFAULT, help="optional save-state name")
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

    try:
        targets = resolve_target(args.target)
    except ValueError as error:
        parser.error(str(error))

    for platform, game in targets:
        print(f"Launching {platform}: {game}", flush=True)
        try:
            run_game(game, args.state, show_obs=args.show_obs)
        except (FileNotFoundError, KeyError, ValueError) as error:
            if len(targets) == 1:
                parser.error(str(error))
            print(f"Could not launch {platform}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

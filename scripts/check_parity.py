#!/usr/bin/env python3
"""Thin development wrapper around TurboBench's owned parity profiles."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROFILES = ("supermario/canonical-v2", "breakout/start-v2")
PROVIDER = "env-stableretro-turbo"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("all", *PROFILES), default="all", nargs="?")
    parser.add_argument("--turbobench", default="turbobench")
    parser.add_argument("--python", default="3.14", dest="python_minor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    for profile in profiles:
        candidate = (
            f"{PROVIDER}@artifact:{args.wheel.resolve()}"
            if args.wheel
            else f"{PROVIDER}@checkout:{root}"
        )
        command = [
            args.turbobench,
            "parity",
            profile,
            "--candidate",
            candidate,
            "--output",
            str(args.output / profile.replace("/", "-")),
            "--python",
            args.python_minor,
        ]
        if not args.wheel:
            command.extend(("--allow-dirty", "--quick"))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

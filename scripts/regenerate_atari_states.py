#!/usr/bin/env python3
"""Migrate curated Atari Start states to the vendored Stella core."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
from pathlib import Path

import stable_retro as retro
from stable_retro.stella_state import migrate_legacy_state


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "stable_retro" / "data" / "stable"


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate(*, update_imports: bool, legacy_ref: str) -> int:
    count = 0
    for game_dir in sorted(DATA_ROOT.glob("*-Atari2600-v0")):
        game = game_dir.name
        expected_sha = (game_dir / "rom.sha").read_text().strip()
        try:
            rom_path = Path(
                retro.data.get_romfile_path(game, retro.data.Integrations.STABLE),
            )
        except FileNotFoundError:
            continue
        if _sha1(rom_path) != expected_sha:
            raise RuntimeError(f"ROM hash mismatch for {game}: {rom_path}")

        old_state = (game_dir / "Start.state").read_bytes()
        legacy_state = gzip.decompress(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{legacy_ref}:stable_retro/data/stable/{game}/Start.state",
                ],
                cwd=REPO_ROOT,
            ),
        )
        emulator = retro.RetroEmulator(str(rom_path))
        migrated_state = migrate_legacy_state(emulator, legacy_state)
        new_state = gzip.compress(migrated_state, compresslevel=9, mtime=0)
        del emulator
        (game_dir / "Start.state").write_bytes(new_state)

        if update_imports:
            imported_state = rom_path.parent / "Start.state"
            if imported_state.resolve() != (game_dir / "Start.state").resolve():
                if imported_state.read_bytes() != old_state:
                    raise RuntimeError(
                        f"refusing to overwrite modified imported state: {imported_state}",
                    )
                imported_state.write_bytes(new_state)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-imports",
        action="store_true",
        help="also update matching imported Start.state files beside each ROM",
    )
    parser.add_argument(
        "--legacy-ref",
        default="v1.0.1.post17",
        help="git ref containing the curated legacy Stella states",
    )
    args = parser.parse_args()
    count = regenerate(
        update_imports=args.update_imports,
        legacy_ref=args.legacy_ref,
    )
    print(f"regenerated_atari_states={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate Super Mario Bros NES stable-retro state files."""

from __future__ import annotations

import argparse
import gzip
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAM_OFFSET = 93
RAM_SIZE = 2048
GAME = "SuperMarioBros-Nes-v0"
GAME_DIR = pathlib.Path("stable_retro/data/stable/SuperMarioBros-Nes-v0")

LEVELS = {
    "Level1-1": (0, 0, 0, 0x25),
    "Level1-2": (0, 1, 1, 0x29),
    "Level1-3": (0, 2, 3, 0x26),
    "Level1-4": (0, 3, 4, 0x60),
    "Level2-1": (1, 0, 0, 0x28),
    "Level2-2": (1, 1, 2, 0x01),
    "Level2-3": (1, 2, 3, 0x27),
    "Level2-4": (1, 3, 4, 0x62),
    "Level3-1": (2, 0, 0, 0x24),
    "Level3-2": (2, 1, 1, 0x35),
    "Level3-3": (2, 2, 2, 0x20),
    "Level3-4": (2, 3, 3, 0x63),
    "Level4-1": (3, 0, 0, 0x22),
    "Level4-2": (3, 1, 1, 0x29),
    "Level4-3": (3, 2, 3, 0x2C),
    "Level4-4": (3, 3, 4, 0x61),
    "Level5-1": (4, 0, 0, 0x2A),
    "Level5-2": (4, 1, 1, 0x31),
    "Level5-3": (4, 2, 2, 0x26),
    "Level5-4": (4, 3, 3, 0x62),
    "Level6-1": (5, 0, 0, 0x2E),
    "Level6-2": (5, 1, 1, 0x23),
    "Level6-3": (5, 2, 2, 0x2D),
    "Level6-4": (5, 3, 3, 0x60),
    "Level7-1": (6, 0, 0, 0x33),
    "Level7-2": (6, 1, 2, 0x01),
    "Level7-3": (6, 2, 3, 0x27),
    "Level7-4": (6, 3, 4, 0x64),
    "Level8-1": (7, 0, 0, 0x30),
    "Level8-2": (7, 1, 1, 0x32),
    "Level8-3": (7, 2, 2, 0x21),
    "Level8-4": (7, 3, 3, 0x65),
}

FINAL_EXPECTED = {
    # 1-2 starts on the aboveground pipe-entry strip (area 1 / pointer 0x29)
    # and then transitions into the actual underground cave (area 2 / pointer 0xc0).
    "Level1-2": (0, 1, 2, 0xC0),
}

MIN_SETTLE_FRAMES = {
    "Level1-2": 540,
}

BASE_STATE_OVERRIDES = {
    "Level1-2": "Level1-1",
}

ENTRANCE_CLEANUP = {
    0x0710: 0,  # PlayerEntranceCtrl
    0x074F: 0,
    0x0751: 0,  # EntrancePage
    0x0752: 0,  # AltEntranceControl
    0x075B: 0,  # HalfwayPage
    0x0762: 0,  # OffScr_HalfwayPage
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("levels", nargs="+", help="Level names, for example Level2-2")
    parser.add_argument("--output-dir", type=pathlib.Path, default=GAME_DIR)
    parser.add_argument("--screens-dir", type=pathlib.Path)
    parser.add_argument("--base-state", default="Level2-1")
    parser.add_argument("--settle-frames", type=int, default=260)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_base_state(game_dir: pathlib.Path, base_state: str) -> bytes:
    path = game_dir / f"{base_state.removesuffix('.state')}.state"
    if not path.exists():
        raise FileNotFoundError(f"base state not found: {path}")
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < RAM_OFFSET + RAM_SIZE:
        raise ValueError(f"state too small to contain NES RAM block: {path}")
    return raw


def patch_level(raw_state: bytes, level_name: str) -> bytes:
    world, level, area, pointer = LEVELS[level_name]
    area_type = (pointer & 0x60) >> 5
    raw = bytearray(raw_state)

    def put(addr: int, value: int) -> None:
        raw[RAM_OFFSET + addr] = value & 0xFF

    put(0x0750, pointer)
    put(0x074E, area_type)
    put(0x075C, level)
    put(0x075F, world)
    put(0x0760, area)
    put(0x0763, level)
    put(0x0766, world)
    put(0x0767, area)
    put(0x0770, 1)
    put(0x0772, 0)
    put(0x000E, 0)
    put(0x0704, 1 if area_type == 0 else 0)
    put(0x0744, 0)
    put(0x00FB, 0)
    for addr, value in ENTRANCE_CLEANUP.items():
        put(addr, value)
    return bytes(raw)


def save_png(screen, path: pathlib.Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for --screens-dir") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(screen).save(path)


def validate_level_name(name: str) -> str:
    if not re.fullmatch(r"Level[1-8]-[1-4]", name):
        raise argparse.ArgumentTypeError(f"not a normal SMB level name: {name}")
    if name not in LEVELS:
        raise argparse.ArgumentTypeError(f"no area mapping for {name}")
    return name


def main() -> int:
    args = parse_args()
    levels = [validate_level_name(level) for level in args.levels]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache = {}

    import stable_retro

    for level_name in levels:
        base_state = BASE_STATE_OVERRIDES.get(level_name, args.base_state)
        if base_state not in base_cache:
            base_cache[base_state] = load_base_state(GAME_DIR, base_state)
        base_raw = base_cache[base_state]
        out_path = output_dir / f"{level_name}.state"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path}", file=sys.stderr)
            continue

        env = stable_retro.make(game=GAME, state=stable_retro.State.NONE, render_mode="rgb_array")
        try:
            env.reset()
            env.em.set_state(patch_level(base_raw, level_name))
            settle_frames = max(args.settle_frames, MIN_SETTLE_FRAMES.get(level_name, 0))
            for _ in range(settle_frames):
                env.em.step()
            raw = env.em.get_state()
            ram = raw[RAM_OFFSET : RAM_OFFSET + RAM_SIZE]
            expected_world, expected_level, expected_area, expected_pointer = FINAL_EXPECTED.get(
                level_name,
                LEVELS[level_name],
            )
            if ram[0x075F] != expected_world or ram[0x075C] != expected_level:
                raise RuntimeError(
                    f"{level_name} settled to world={ram[0x075F]} level={ram[0x075C]}"
                )
            if ram[0x0760] != expected_area:
                raise RuntimeError(f"{level_name} settled to area={ram[0x0760]}")
            if ram[0x0750] != expected_pointer:
                raise RuntimeError(f"{level_name} settled to pointer=0x{ram[0x0750]:02x}")
            if ram[0x0770] != 1 or ram[0x0772] < 3:
                raise RuntimeError(
                    f"{level_name} did not settle into gameplay: op={ram[0x0770]} task={ram[0x0772]}"
                )
            with gzip.open(out_path, "wb", compresslevel=9) as fh:
                fh.write(raw)
            if args.screens_dir:
                save_png(env.em.get_screen(), args.screens_dir / f"{level_name}.png")
            print(
                f"{level_name}: wrote {out_path} "
                f"world={ram[0x075F] + 1} level={ram[0x075C] + 1} area={ram[0x0760]} "
                f"raw={len(raw)} gzip={out_path.stat().st_size}"
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

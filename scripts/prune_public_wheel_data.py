#!/usr/bin/env python3
"""Prune unsupported platform data from public stable-retro wheel artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


DEFAULT_PLATFORMS = ("GameBoy", "Nes", "Snes", "Genesis", "Sms", "SCD")


def data_dir_platform(name: str) -> str:
    parts = name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-1].startswith("v"):
        return parts[-2]
    return name.rsplit("-", 1)[-1]


def should_prune(path: Path, root: Path, platforms: set[str]) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 5:
        return False
    if parts[0] != "env_stableretro_turbo" or parts[1] != "data":
        return False
    if parts[2] not in {"stable", "contrib", "experimental"}:
        return False
    return data_dir_platform(parts[3]) not in platforms


def file_hash(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", str(path.stat().st_size)


def refresh_record(root: Path) -> None:
    records = list(root.glob("*.dist-info/RECORD"))
    if len(records) != 1:
        raise RuntimeError(f"expected one RECORD file, found {len(records)}")

    record = records[0]
    rows: list[list[str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path == record:
            rows.append([rel, "", ""])
            continue
        digest, size = file_hash(path)
        rows.append([rel, digest, size])

    with record.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def repack_wheel(root: Path, wheel: Path) -> None:
    tmp_wheel = wheel.with_suffix(".whl.tmp")
    with zipfile.ZipFile(
        tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo.from_file(path, rel)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    tmp_wheel.replace(wheel)


def prune_wheel(wheel: Path, out_dir: Path | None, platforms: set[str]) -> Path:
    wheel = wheel.resolve()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / wheel.name
        shutil.copy2(wheel, target)
    else:
        target = wheel

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wheel"
        root.mkdir()
        with zipfile.ZipFile(target) as zf:
            zf.extractall(root)

        removed = 0
        removed_bytes = 0
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if should_prune(path, root, platforms):
                removed += 1
                removed_bytes += path.stat().st_size
                path.unlink()

        for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass

        refresh_record(root)
        repack_wheel(root, target)

    print(f"{target}: pruned {removed} files / {removed_bytes} raw bytes")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--platforms",
        default=",".join(DEFAULT_PLATFORMS),
        help="Comma-separated platform names to keep",
    )
    args = parser.parse_args()

    platforms = {platform.strip() for platform in args.platforms.split(",") if platform.strip()}
    for wheel in args.wheels:
        prune_wheel(wheel, args.out_dir, platforms)


if __name__ == "__main__":
    main()

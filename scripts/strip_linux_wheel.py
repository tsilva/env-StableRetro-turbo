#!/usr/bin/env python3
"""Strip ELF shared objects inside Linux wheels and refresh RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


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


def strip_wheel(wheel: Path, strip_cmd: str, out_dir: Path | None) -> Path:
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

        total_saved = 0
        for so in sorted(root.rglob("*.so")):
            before = so.stat().st_size
            subprocess.run([strip_cmd, "--strip-unneeded", str(so)], check=True)
            total_saved += before - so.stat().st_size

        refresh_record(root)
        repack_wheel(root, target)

    print(f"{target}: stripped {total_saved} raw bytes")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--strip-cmd", default="strip")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    for wheel in args.wheels:
        strip_wheel(wheel, args.strip_cmd, args.out_dir)


if __name__ == "__main__":
    main()

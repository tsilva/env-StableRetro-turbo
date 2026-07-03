#!/usr/bin/env python3
"""Deterministic helpers for stable-retro-turbo release wheel builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = REPO_ROOT / "stable_retro" / "VERSION.txt"
PYTHON = REPO_ROOT / ".venv314" / "bin" / "python"
PACKAGE_NAME = "stable-retro-turbo"

PUBLIC_CORES = ("gambatte", "fceumm", "snes9x", "genesis_plus_gx")
PUBLIC_DATA_PLATFORMS = "GameBoy,Nes,Snes,Genesis,Sms,SCD"
MACOS_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_CORES=gb;nes;snes;genesis "
    "-DBUILD_TESTS=OFF "
    "-DENABLE_CAPNPROTO=OFF "
    "-DSTABLE_RETRO_USE_SYSTEM_LIBZIP=OFF"
)
LINUX_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_MANYLINUX=ON "
    "-DBUILD_CORES=gb;nes;snes;genesis "
    "-DBUILD_TESTS=OFF "
    "-DENABLE_CAPNPROTO=OFF "
    "-DBUILD_N64=OFF"
)

IGNORED_DIR_NAMES_ANYWHERE = {
    ".git",
    ".venv314",
    "__pycache__",
    ".pytest_cache",
    "CMakeFiles",
}
IGNORED_ROOT_DIR_NAMES = {"build", "dist", "env"}
IGNORED_FILE_NAMES = {"CMakeCache.txt"}
IGNORED_FILE_SUFFIXES = {".o", ".a", ".so", ".dylib", ".d"}
ROM_PAYLOAD_NAMES = {
    "rom.nes",
    "rom.sfc",
    "rom.smc",
    "rom.gb",
    "rom.gbc",
    "rom.md",
    "rom.gen",
    "rom.sms",
    "rom.bin",
}
GAME_PAYLOAD_SUFFIXES = (".nes", ".sfc", ".smc", ".gb", ".gbc", ".gen", ".sms", ".bin")
VERSION_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?:\.post(?P<post>\d+))?$")


def read_version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip()


def check_version(args: argparse.Namespace) -> None:
    version = read_version()
    parse_version(version)
    package_code, package_output = run_capture([str(PYTHON), "setup.py", "--name"])
    result = {
        "package": package_output,
        "version": version,
    }
    print(json.dumps(result, indent=2))
    failures = []
    if package_code != 0:
        failures.append(f"setup.py --name failed: {package_output}")
    if package_output != PACKAGE_NAME:
        failures.append(f"package name is {package_output!r}, expected {PACKAGE_NAME!r}")
    if args.version is not None and version != args.version:
        failures.append(f"expected version {args.version!r}, saw {version!r}")
    if failures:
        raise SystemExit("; ".join(failures))


def parse_version(version: str) -> tuple[str, int]:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"unsupported version format: {version!r}")
    return match.group("base"), int(match.group("post") or 0)


def next_post_version(version: str) -> str:
    base, post = parse_version(version)
    return f"{base}.post{post + 1}"


def post_number(version: str) -> int:
    _, post = parse_version(version)
    return post


def expected_wheelhouse(version: str, platform_name: str) -> Path:
    suffix = "repaired" if platform_name == "macos" else "linux"
    return REPO_ROOT / f"wheelhouse-post{post_number(version)}-{suffix}"


def expected_macos_wheel(version: str) -> Path:
    return expected_wheelhouse(version, "macos") / (
        f"stable_retro_turbo-{version}-cp314-cp314-macosx_14_0_arm64.whl"
    )


def expected_linux_wheel(version: str) -> Path:
    return expected_wheelhouse(version, "linux") / (
        f"stable_retro_turbo-{version}-cp314-cp314-"
        "manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl"
    )


def is_under(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def is_rom_payload(rel: Path) -> bool:
    parts = rel.parts
    return is_under(parts, ("stable_retro", "data")) and rel.name.lower() in ROM_PAYLOAD_NAMES


def should_ignore(rel: Path) -> bool:
    parts = rel.parts
    if any(part in IGNORED_DIR_NAMES_ANYWHERE for part in parts):
        return True
    if len(parts) == 1 and rel.name in IGNORED_ROOT_DIR_NAMES:
        return True
    if rel.name in IGNORED_FILE_NAMES:
        return True
    if rel.name.startswith("wheelhouse"):
        return True
    if rel.suffix in IGNORED_FILE_SUFFIXES:
        return True
    return is_rom_payload(rel)


def copy_clean_tree(destination: Path, *, force: bool = False) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; pass --force to replace it")
        shutil.rmtree(destination)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            rel = (base / name).relative_to(REPO_ROOT)
            if should_ignore(rel):
                ignored.add(name)
        return ignored

    shutil.copytree(REPO_ROOT, destination, symlinks=True, ignore=ignore)


def find_contamination(root: Path) -> dict[str, list[str]]:
    compiled: list[str] = []
    rom_payloads: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.suffix in IGNORED_FILE_SUFFIXES:
            compiled.append(rel.as_posix())
        if is_rom_payload(rel):
            rom_payloads.append(rel.as_posix())
    return {"compiled_artifacts": compiled, "rom_payloads": rom_payloads}


def fail_on_contamination(root: Path) -> None:
    contamination = find_contamination(root)
    failures = {key: value for key, value in contamination.items() if value}
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit(f"{root} is not a clean release source copy")


def shell_quote(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def run_capture(args_list: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args_list,
            check=False,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def env_lines(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shell_quote(value)}" for key, value in env.items())


def macos_env() -> dict[str, str]:
    return {
        "MACOSX_DEPLOYMENT_TARGET": "14.0",
        "ARCHFLAGS": "-arch arm64",
        "CMAKE_ARGS": MACOS_CMAKE_ARGS,
        "STABLE_RETRO_PUBLIC_CORES": ",".join(PUBLIC_CORES),
        "STABLE_RETRO_PUBLIC_DATA_PLATFORMS": PUBLIC_DATA_PLATFORMS,
    }


def linux_env() -> dict[str, str]:
    return {
        "CIBW_BUILD": "cp314-manylinux_x86_64",
        "CIBW_ARCHS_LINUX": "x86_64",
    }


def prepare_sources(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    post = post_number(version)
    root = args.root or Path(
        tempfile.mkdtemp(prefix=f"stable-retro-turbo-post{post}-builds.", dir="/private/tmp")
    )
    root = root.resolve()
    macos_src = root / "macos-src"
    linux_src = root / "linux-src-clean"
    copy_clean_tree(macos_src, force=args.force)
    copy_clean_tree(linux_src, force=args.force)
    fail_on_contamination(linux_src)
    result = {
        "version": version,
        "post": post,
        "root": str(root),
        "macos_src": str(macos_src),
        "linux_src_clean": str(linux_src),
        "repo": str(REPO_ROOT),
        "python": str(PYTHON),
        "macos_wheelhouse": str(expected_wheelhouse(version, "macos")),
        "linux_wheelhouse": str(expected_wheelhouse(version, "linux")),
    }
    print(json.dumps(result, indent=2))


def bump_version(args: argparse.Namespace) -> None:
    current = read_version()
    target = args.to or next_post_version(current)
    parse_version(target)
    if args.write:
        VERSION_PATH.write_text(f"{target}\n", encoding="utf-8")
    print(target)


def check_pypi(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    parse_version(version)
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(json.dumps({"package": PACKAGE_NAME, "exists": False, "version_exists": False}, indent=2))
            return
        raise
    releases = data.get("releases", {})
    exists = version in releases and bool(releases[version])
    print(json.dumps({"package": PACKAGE_NAME, "version": version, "version_exists": exists}, indent=2))
    if exists:
        raise SystemExit(f"{PACKAGE_NAME} {version} already exists on PyPI")


def build_commands(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    macos_src = Path(args.macos_src) if args.macos_src else Path("<macos-src>")
    linux_src = Path(args.linux_src) if args.linux_src else Path("<linux-src-clean>")
    macos_wheelhouse = expected_wheelhouse(version, "macos")
    linux_wheelhouse = expected_wheelhouse(version, "linux")
    print("# macOS arm64")
    print(f"cd {shell_quote(macos_src)}")
    print(
        f"{env_lines(macos_env())} {shell_quote(PYTHON)} "
        "setup.py bdist_wheel --plat-name macosx_14_0_arm64"
    )
    print(
        f"{shell_quote(PYTHON)} -m delocate.cmd.delocate_wheel --require-archs arm64 "
        f"-w {shell_quote(macos_wheelhouse)} -v dist/*.whl"
    )
    print(
        f"{shell_quote(PYTHON)} scripts/strip_macos_wheel.py "
        f"{shell_quote(expected_macos_wheel(version))}"
    )
    print()
    print("# Linux manylinux")
    print(f"cd {shell_quote(linux_src)}")
    print(
        f"{env_lines(linux_env())} {shell_quote(PYTHON)} -m cibuildwheel "
        f"--platform linux --output-dir {shell_quote(linux_wheelhouse)}"
    )


def clean_output_paths(version: str, platform_name: str) -> None:
    wheelhouse = expected_wheelhouse(version, platform_name)
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    if platform_name == "macos":
        dist = REPO_ROOT / "dist"
        build = REPO_ROOT / "build"
        for path in (dist, build):
            if path.exists():
                shutil.rmtree(path)


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    parse_version(version)
    clean_output_paths(version, args.platform)
    if args.platform == "macos":
        env = os.environ.copy()
        env.update(macos_env())
        env["PATH"] = f"{PYTHON.parent}{os.pathsep}{env.get('PATH', '')}"
        run(
            [
                str(PYTHON),
                "setup.py",
                "bdist_wheel",
                "--plat-name",
                "macosx_14_0_arm64",
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        raw_wheels = sorted((REPO_ROOT / "dist").glob("*.whl"))
        if len(raw_wheels) != 1:
            raise SystemExit(f"expected one raw macOS wheel, found {len(raw_wheels)}")
        run(
            [
                str(PYTHON),
                "-m",
                "delocate.cmd.delocate_wheel",
                "--require-archs",
                "arm64",
                "-w",
                str(expected_wheelhouse(version, "macos")),
                "-v",
                str(raw_wheels[0]),
            ]
        )
        run([str(PYTHON), "scripts/strip_macos_wheel.py", str(expected_macos_wheel(version))], cwd=REPO_ROOT)
        print(expected_macos_wheel(version))
    elif args.platform == "linux":
        env = os.environ.copy()
        env.update(linux_env())
        env["PATH"] = f"{PYTHON.parent}{os.pathsep}{env.get('PATH', '')}"
        run(
            [
                str(PYTHON),
                "-m",
                "cibuildwheel",
                "--platform",
                "linux",
                "--output-dir",
                str(expected_wheelhouse(version, "linux")),
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        print(expected_linux_wheel(version))
    else:  # pragma: no cover - argparse choices guard this.
        raise ValueError(args.platform)


def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def audit_wheel(wheel: Path, version: str, platform_name: str) -> dict[str, object]:
    names = wheel_names(wheel)
    if platform_name == "macos":
        expected_name = expected_macos_wheel(version).name
        expected_extension = "stable_retro/_retro.cpython-314-darwin.so"
        core_suffix = ".dylib"
    elif platform_name == "linux":
        expected_name = expected_linux_wheel(version).name
        expected_extension = "stable_retro/_retro.cpython-314-x86_64-linux-gnu.so"
        core_suffix = ".so"
    else:
        raise ValueError(f"unknown platform: {platform_name}")

    rom_payloads = []
    for name in names:
        if not name.startswith("stable_retro/data/"):
            continue
        lower_name = name.lower()
        if Path(name).name.lower() in ROM_PAYLOAD_NAMES or lower_name.endswith(
            GAME_PAYLOAD_SUFFIXES
        ):
            rom_payloads.append(name)
    checks = {
        "expected_filename": wheel.name == expected_name,
        "version_in_filename": version in wheel.name,
        "expected_extension": expected_extension in names,
        "no_cp311_extension": not any("cp311" in name for name in names),
        "all_public_cores": all(
            f"stable_retro/cores/{core}_libretro{core_suffix}" in names
            for core in PUBLIC_CORES
        ),
        "all_public_core_json": all(
            f"stable_retro/cores/{core}.json" in names for core in PUBLIC_CORES
        ),
        "no_rom_payloads": not rom_payloads,
        "has_retro_vec_env_source": "stable_retro/vec_env.py" in names,
    }
    with zipfile.ZipFile(wheel) as zf:
        init = zf.read("stable_retro/__init__.py").decode("utf-8")
        vec_env = zf.read("stable_retro/vec_env.py").decode("utf-8")
    checks.update(
        {
            "exports_retro_vec_env": "RetroVecEnv" in init,
            "no_stable_retro_native_vec_env_export": "StableRetroNativeVecEnv" not in init,
            "class_retro_vec_env": "class RetroVecEnv" in vec_env,
            "no_stable_retro_native_vec_env_class": "StableRetroNativeVecEnv" not in vec_env,
        }
    )
    return {
        "wheel": str(wheel),
        "platform": platform_name,
        "checks": checks,
        "rom_payloads": rom_payloads,
    }


def assert_audit_passed(results: list[dict[str, object]]) -> None:
    failures: dict[str, list[str]] = {}
    for result in results:
        checks = result["checks"]
        assert isinstance(checks, dict)
        failed = [key for key, value in checks.items() if not value]
        if failed:
            failures[str(result["wheel"])] = failed
    if failures:
        print(json.dumps(results, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failures}")


def audit_wheels(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    macos_wheel = args.macos_wheel or expected_macos_wheel(version)
    linux_wheel = args.linux_wheel or expected_linux_wheel(version)
    results = [
        audit_wheel(macos_wheel, version, "macos"),
        audit_wheel(linux_wheel, version, "linux"),
    ]
    assert_audit_passed(results)
    print(json.dumps(results, indent=2))


def run(args_list: list[str], **kwargs: object) -> None:
    print("+", " ".join(shell_quote(arg) for arg in args_list))
    subprocess.run(args_list, check=True, **kwargs)


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = args.wheel.absolute()
    python = args.python.absolute()
    uv_env = os.environ.copy()
    uv_env.setdefault("UV_CACHE_DIR", "/private/tmp/uv-cache-stable-retro-build")
    with tempfile.TemporaryDirectory(prefix="stable-retro-macos-wheel-smoke.", dir="/private/tmp") as tmp:
        target = Path(tmp)
        if args.installer == "uv":
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheel),
                ],
                env=uv_env,
            )
        elif args.installer == "pip":
            run([str(python), "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)])
        else:
            try:
                run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-deps",
                        "--target",
                        str(target),
                        str(wheel),
                    ],
                    env=uv_env,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                run([str(python), "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)])

        code = """
import stable_retro
from stable_retro import _retro
print(stable_retro.__file__)
print(_retro.__file__)
assert stable_retro.__file__.startswith({target!r})
assert hasattr(_retro, "NativeVectorEnv")
assert hasattr(stable_retro, "RetroVecEnv")
assert not hasattr(stable_retro, "StableRetroNativeVecEnv")
""".format(target=str(target))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stable-retro")
        run([str(python), "-c", code], cwd="/private/tmp", env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_note(version: str) -> str:
    return (
        "GitHub trusted publishing handles the PyPI upload for tag pushes.\n"
        f"Push tag v{version} after the release commit is ready; do not upload these wheels with local Twine."
    )


def final_check(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    macos_wheel = expected_macos_wheel(version)
    linux_wheel = expected_linux_wheel(version)
    results = [
        audit_wheel(macos_wheel, version, "macos"),
        audit_wheel(linux_wheel, version, "linux"),
    ]
    assert_audit_passed(results)
    run([str(PYTHON), "-m", "twine", "check", str(macos_wheel), str(linux_wheel)])
    hashes = {str(wheel): sha256(wheel) for wheel in (macos_wheel, linux_wheel)}
    print(json.dumps({"audits": results, "sha256": hashes}, indent=2))
    print()
    print(publish_note(version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-version", help="Validate package name and version")
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    bump = subparsers.add_parser("bump-version", help="Print or write the next post version")
    bump.add_argument("--to", help="Set an explicit target version instead of incrementing")
    bump.add_argument("--write", action="store_true", help="Write the target to stable_retro/VERSION.txt")
    bump.set_defaults(func=bump_version)

    pypi = subparsers.add_parser("check-pypi", help="Check whether a PyPI version is still unused")
    pypi.add_argument("--version")
    pypi.set_defaults(func=check_pypi)

    prepare = subparsers.add_parser("prepare-sources", help="Create clean macOS/Linux source copies")
    prepare.add_argument("--version")
    prepare.add_argument("--root", type=Path)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=prepare_sources)

    commands = subparsers.add_parser("build-commands", help="Print platform build commands")
    commands.add_argument("--version")
    commands.add_argument("--macos-src")
    commands.add_argument("--linux-src")
    commands.set_defaults(func=build_commands)

    build = subparsers.add_parser("build-platform", help="Build one release wheel platform")
    build.add_argument("--platform", choices=("macos", "linux"), required=True)
    build.add_argument("--version")
    build.set_defaults(func=build_platform)

    audit = subparsers.add_parser("audit-wheels", help="Audit macOS and Linux wheel contents")
    audit.add_argument("--version")
    audit.add_argument("--macos-wheel", type=Path)
    audit.add_argument("--linux-wheel", type=Path)
    audit.set_defaults(func=audit_wheels)

    smoke = subparsers.add_parser("smoke-wheel", help="Install and import-test a wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--python", type=Path, default=PYTHON)
    smoke.add_argument("--installer", choices=("auto", "uv", "pip"), default="auto")
    smoke.set_defaults(func=smoke_wheel)

    smoke_macos = subparsers.add_parser("smoke-macos-wheel", help="Install and import-test a macOS wheel")
    smoke_macos.add_argument("wheel", type=Path)
    smoke_macos.add_argument("--python", type=Path, default=PYTHON)
    smoke_macos.add_argument("--installer", choices=("auto", "uv", "pip"), default="auto")
    smoke_macos.set_defaults(func=smoke_wheel)

    final = subparsers.add_parser("final-check", help="Audit wheels, run twine check, hash, and print publishing handoff")
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

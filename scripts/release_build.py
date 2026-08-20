#!/usr/bin/env python3
"""Deterministic helpers for env-StableRetro-turbo release builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = REPO_ROOT / "env_stableretro_turbo" / "VERSION.txt"
PYTHON = REPO_ROOT / ".venv314" / "bin" / "python"
PACKAGE_NAME = "env-stableretro-turbo"
PYTHON_TAGS = ("cp314",)
RELEASE_PLATFORMS = (
    "macos-arm64",
    "linux-x86_64",
)

PUBLIC_CORES = (
    "gambatte",
    "fceumm",
    "snes9x",
    "genesis_plus_gx",
    "stella",
    "mgba",
    "picodrive",
    "mednafen_saturn",
    "melonds",
)
PUBLIC_DATA_PLATFORMS = (
    "GameBoy,Nes,Snes,Genesis,Sms,SCD,Atari2600,GbAdvance,32x,Saturn,NintendoDs"
)
PUBLIC_CORE_SOURCE_DIRS = frozenset(
    {
        "32x",
        "atari2600",
        "ds",
        "gb",
        "gba",
        "genesis",
        "nes",
        "saturn",
        "snes",
    },
)
KNOWN_CORE_SOURCE_DIRS = PUBLIC_CORE_SOURCE_DIRS | {
    "fbneo",
    "flycast",
    "n64",
    "pce",
}
MACOS_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release "
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache "
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
    "-DBUILD_CORES=gb;nes;snes;genesis;atari2600;gba;32x;saturn;ds "
    "-DBUILD_TESTS=OFF "
    "-DENABLE_CAPNPROTO=OFF "
    "-DENV_STABLERETRO_TURBO_USE_SYSTEM_LIBZIP=OFF"
)
LINUX_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release "
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache "
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
    "-DBUILD_MANYLINUX=ON "
    "-DBUILD_CORES=gb;nes;snes;genesis;atari2600;gba;32x;saturn;ds "
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
IGNORED_ROOT_DIR_NAMES = {".ccache", "build", "dist", "env"}
IGNORED_FILE_NAMES = {"CMakeCache.txt"}
IGNORED_FILE_SUFFIXES = {".o", ".a", ".so", ".dylib", ".dll", ".pyd", ".d"}
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
    "rom.a26",
}
GAME_PAYLOAD_SUFFIXES = (
    ".nes",
    ".sfc",
    ".smc",
    ".gb",
    ".gbc",
    ".gen",
    ".sms",
    ".bin",
    ".a26",
)
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
        failures.append(
            f"package name is {package_output!r}, expected {PACKAGE_NAME!r}",
        )
    if args.version is not None and version != args.version:
        failures.append(f"expected version {args.version!r}, saw {version!r}")
    if failures:
        raise SystemExit("; ".join(failures))


def parse_version(version: str) -> tuple[str, int]:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"unsupported version format: {version!r}")
    return match.group("base"), int(match.group("post") or 0)


def version_sort_key(version: str) -> tuple[int, int, int, int]:
    base, post = parse_version(version)
    major, minor, patch = (int(part) for part in base.split("."))
    return major, minor, patch, post


def next_post_version(version: str) -> str:
    base, post = parse_version(version)
    return f"{base}.post{post + 1}"


def post_number(version: str) -> int:
    _, post = parse_version(version)
    return post


def expected_wheelhouse(version: str, platform_name: str) -> Path:
    suffixes = {
        "macos-arm64": "repaired",
        "linux-x86_64": "linux",
    }
    try:
        suffix = suffixes[platform_name]
    except KeyError as exc:
        raise ValueError(f"unknown platform: {platform_name}") from exc
    return REPO_ROOT / f"wheelhouse-post{post_number(version)}-{suffix}"


def expected_macos_wheels(version: str) -> list[Path]:
    return [
        expected_wheelhouse(version, "macos-arm64")
        / f"env_stableretro_turbo-{version}-{tag}-{tag}-macosx_14_0_arm64.whl"
        for tag in PYTHON_TAGS
    ]


def expected_linux_wheels(version: str) -> list[Path]:
    return [
        expected_wheelhouse(version, "linux-x86_64")
        / (
            f"env_stableretro_turbo-{version}-{tag}-{tag}-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        )
        for tag in PYTHON_TAGS
    ]


def expected_wheels(version: str, platform_name: str) -> list[Path]:
    if platform_name == "macos-arm64":
        return expected_macos_wheels(version)
    if platform_name == "linux-x86_64":
        return expected_linux_wheels(version)
    raise ValueError(f"unknown platform: {platform_name}")


def expected_sdist(version: str) -> Path:
    return REPO_ROOT / "dist" / f"env_stableretro_turbo-{version}.tar.gz"


def is_under(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def is_rom_payload(rel: Path) -> bool:
    parts = rel.parts
    return (
        is_under(parts, ("env_stableretro_turbo", "data"))
        and rel.name.lower() in ROM_PAYLOAD_NAMES
    )


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
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace it",
            )
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


def data_dir_platform(name: str) -> str:
    parts = name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-1].startswith("v"):
        return parts[-2]
    return name.rsplit("-", 1)[-1]


def prune_sdist_tree(root: Path) -> None:
    cores = root / "cores"
    for path in cores.iterdir():
        if path.is_dir() and path.name not in PUBLIC_CORE_SOURCE_DIRS:
            shutil.rmtree(path)

    public_platforms = frozenset(PUBLIC_DATA_PLATFORMS.split(","))
    data = root / "env_stableretro_turbo" / "data"
    for collection in ("stable", "experimental", "contrib"):
        collection_root = data / collection
        if not collection_root.is_dir():
            continue
        for path in collection_root.iterdir():
            if path.is_dir() and data_dir_platform(path.name) not in public_platforms:
                shutil.rmtree(path)

    libzip_regress = root / "third-party" / "libzip" / "regress"
    if libzip_regress.exists():
        shutil.rmtree(libzip_regress)
    for unused_dependency in ("capnproto", "gtest"):
        path = root / "third-party" / unused_dependency
        if path.exists():
            shutil.rmtree(path)
    pybind11 = root / "third-party" / "pybind11"
    if pybind11.is_dir():
        for unused_pybind11_path in ("docs", "pybind11", "tests", "tools"):
            path = pybind11 / unused_pybind11_path
            if path.exists():
                shutil.rmtree(path)
    for unused_core_asset in (
        root / "cores" / "gba" / "cinema",
        root / "cores" / "gba" / "doc",
        root / "cores" / "gba" / "res",
        root / "cores" / "gba" / "tools",
        root / "cores" / "genesis" / "builds",
        root / "cores" / "genesis" / "gcw0",
        root / "cores" / "genesis" / "gx",
        root / "cores" / "genesis" / "psp2",
        root / "cores" / "genesis" / "sdl",
        root / "cores" / "32x" / "platform" / "gizmondo",
        root / "cores" / "32x" / "platform" / "gp2x",
        root / "cores" / "32x" / "platform" / "opendingux",
        root / "cores" / "32x" / "platform" / "pandora",
        root / "cores" / "32x" / "platform" / "psp",
        root / "cores" / "32x" / "platform" / "win32",
        root / "cores" / "32x" / "tools",
        root / "cores" / "ds" / ".github",
        root / "cores" / "ds" / "icon",
        root / "cores" / "ds" / "tools",
    ):
        if unused_core_asset.exists():
            shutil.rmtree(unused_core_asset)
    gba_platform = root / "cores" / "gba" / "src" / "platform"
    if gba_platform.is_dir():
        for path in gba_platform.iterdir():
            if path.is_dir() and path.name not in {"libretro", "posix"}:
                shutil.rmtree(path)
    for core_root in (root / "cores").iterdir():
        if not core_root.is_dir():
            continue
        for path in core_root.rglob("*"):
            if path.is_file() and path.suffix in IGNORED_FILE_SUFFIXES:
                path.unlink()


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


def release_temp_root() -> Path:
    root = Path("/private/tmp")
    if not root.exists():
        root = Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def compiler_cache_dir(root: Path = REPO_ROOT) -> Path:
    return root / ".ccache"


def macos_env(root: Path = REPO_ROOT) -> dict[str, str]:
    return {
        "MACOSX_DEPLOYMENT_TARGET": "14.0",
        "ARCHFLAGS": "-arch arm64",
        "CCACHE_DIR": str(compiler_cache_dir(root)),
        "CCACHE_BASEDIR": str(root),
        "CCACHE_COMPILERCHECK": "content",
        "CCACHE_MAXSIZE": "2G",
        "CMAKE_ARGS": MACOS_CMAKE_ARGS,
        "ENV_STABLERETRO_TURBO_PUBLIC_CORES": ",".join(PUBLIC_CORES),
        "ENV_STABLERETRO_TURBO_PUBLIC_DATA_PLATFORMS": PUBLIC_DATA_PLATFORMS,
    }


def linux_env(root: Path = REPO_ROOT) -> dict[str, str]:
    cache_dir = compiler_cache_dir(root).resolve()
    return {
        "CIBW_BUILD": " ".join(f"{tag}-manylinux_x86_64" for tag in PYTHON_TAGS),
        "CIBW_ARCHS_LINUX": "x86_64",
        "CIBW_CONTAINER_ENGINE": (f"docker; create_args: --volume={cache_dir}:/ccache"),
    }


def prepare_sources(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    post = post_number(version)
    root = args.root or Path(
        tempfile.mkdtemp(
            prefix=f"env-stableretro-turbo-post{post}-builds.",
            dir=release_temp_root(),
        ),
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
        "macos_wheelhouse": str(expected_wheelhouse(version, "macos-arm64")),
        "linux_wheelhouse": str(expected_wheelhouse(version, "linux-x86_64")),
    }
    print(json.dumps(result, indent=2))


def bump_version(args: argparse.Namespace) -> None:
    current = read_version()
    target = args.to or next_post_version(current)
    parse_version(target)
    if args.write:
        VERSION_PATH.write_text(f"{target}\n", encoding="utf-8")
    print(target)


def fetch_pypi_project(package: str = PACKAGE_NAME) -> dict[str, object]:
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def release_has_non_yanked_file(files: object) -> bool:
    if not isinstance(files, list):
        return False
    return any(
        isinstance(file, dict) and not file.get("yanked", False) for file in files
    )


def latest_non_yanked_pypi_version(releases: object) -> str | None:
    if not isinstance(releases, dict):
        return None
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for version, files in releases.items():
        if not isinstance(version, str) or not release_has_non_yanked_file(files):
            continue
        try:
            candidates.append((version_sort_key(version), version))
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def check_pypi(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    package = args.package or PACKAGE_NAME
    parse_version(version)
    try:
        data = fetch_pypi_project(package)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                json.dumps(
                    {"package": package, "exists": False, "version_exists": False},
                    indent=2,
                ),
            )
            return
        raise
    releases = data.get("releases", {})
    exists = version in releases and bool(releases[version])
    print(
        json.dumps(
            {"package": package, "version": version, "version_exists": exists},
            indent=2,
        ),
    )
    if exists:
        raise SystemExit(f"{package} {version} already exists on PyPI")


def latest_pypi(args: argparse.Namespace) -> None:
    try:
        data = fetch_pypi_project()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                json.dumps(
                    {
                        "package": PACKAGE_NAME,
                        "exists": False,
                        "latest_non_yanked": None,
                    },
                    indent=2,
                ),
            )
            return
        raise
    latest = latest_non_yanked_pypi_version(data.get("releases"))
    info = data.get("info", {})
    info_version = info.get("version") if isinstance(info, dict) else None
    result = {
        "package": PACKAGE_NAME,
        "exists": True,
        "latest_non_yanked": latest,
        "pypi_info_version": info_version,
    }
    print(json.dumps(result, indent=2))
    if args.fail_if_mismatch and latest != info_version:
        raise SystemExit(
            f"PyPI info.version {info_version!r} does not match latest non-yanked {latest!r}",
        )


def build_commands(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    macos_src = Path(args.macos_src) if args.macos_src else Path("<macos-src>")
    linux_src = Path(args.linux_src) if args.linux_src else Path("<linux-src-clean>")
    macos_wheelhouse = expected_wheelhouse(version, "macos-arm64")
    linux_wheelhouse = expected_wheelhouse(version, "linux-x86_64")
    print("# macOS arm64")
    print(f"cd {shell_quote(macos_src)}")
    print(
        f"{env_lines(macos_env(macos_src))} {shell_quote(PYTHON)} -m cibuildwheel "
        f"--platform macos --output-dir {shell_quote(macos_wheelhouse)}",
    )
    print()
    print("# Linux manylinux")
    print(f"cd {shell_quote(linux_src)}")
    print(
        f"{env_lines(linux_env(linux_src))} {shell_quote(PYTHON)} -m cibuildwheel "
        f"--platform linux --output-dir {shell_quote(linux_wheelhouse)}",
    )


def clean_output_paths(version: str, platform_name: str) -> None:
    wheelhouse = expected_wheelhouse(version, platform_name)
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    if platform_name == "macos-arm64":
        dist = REPO_ROOT / "dist"
        build = REPO_ROOT / "build"
        for path in (dist, build):
            if path.exists():
                shutil.rmtree(path)


def build_platform(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    parse_version(version)
    compiler_cache_dir().mkdir(parents=True, exist_ok=True)
    clean_output_paths(version, args.platform)
    if args.platform == "macos-arm64":
        env = os.environ.copy()
        env.update(macos_env())
        env["PATH"] = f"{PYTHON.parent}{os.pathsep}{env.get('PATH', '')}"
        run(
            [
                str(PYTHON),
                "-m",
                "cibuildwheel",
                "--platform",
                "macos",
                "--output-dir",
                str(expected_wheelhouse(version, "macos-arm64")),
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        for wheel in expected_macos_wheels(version):
            print(wheel)
    elif args.platform == "linux-x86_64":
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
                str(expected_wheelhouse(version, "linux-x86_64")),
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        for wheel in expected_linux_wheels(version):
            print(wheel)
    else:  # pragma: no cover - argparse choices guard this.
        raise ValueError(args.platform)


def build_sdist(args: argparse.Namespace) -> None:
    version = args.version or read_version()
    parse_version(version)
    destination = expected_sdist(version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"env-stableretro-turbo-{version}-sdist.",
        dir=release_temp_root(),
    ) as tmp:
        source = Path(tmp) / "source"
        copy_clean_tree(source)
        prune_sdist_tree(source)
        fail_on_contamination(source)
        env = os.environ.copy()
        env["ENV_STABLERETRO_TURBO_PUBLIC_CORES"] = ",".join(PUBLIC_CORES)
        env["ENV_STABLERETRO_TURBO_PUBLIC_DATA_PLATFORMS"] = PUBLIC_DATA_PLATFORMS
        run(
            [
                str(PYTHON),
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(destination.parent),
            ],
            cwd=source,
            env=env,
        )
    if not destination.is_file():
        raise SystemExit(f"source build did not produce {destination}")
    result = audit_sdist(destination, version)
    assert_sdist_audit_passed(result)
    print(json.dumps(result, indent=2))


def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def wheel_python_tag(wheel: Path) -> str:
    match = re.search(r"-(cp\d+)-cp\d+-", wheel.name)
    if match is None:
        raise ValueError(f"could not infer Python tag from wheel name: {wheel.name}")
    return match.group(1)


def audit_wheel(wheel: Path, version: str, platform_name: str) -> dict[str, object]:
    names = wheel_names(wheel)
    tag = wheel_python_tag(wheel)
    python_abi = tag.replace("cp", "")
    if platform_name == "macos-arm64":
        expected_names = {path.name for path in expected_macos_wheels(version)}
        expected_extension = f"env_stableretro_turbo/_retro.cpython-{python_abi}-darwin.so"
        core_suffix = ".dylib"
    elif platform_name == "linux-x86_64":
        expected_names = {path.name for path in expected_linux_wheels(version)}
        expected_extension = (
            f"env_stableretro_turbo/_retro.cpython-{python_abi}-x86_64-linux-gnu.so"
        )
        core_suffix = ".so"
    else:
        raise ValueError(f"unknown platform: {platform_name}")

    rom_payloads = []
    for name in names:
        if not name.startswith("env_stableretro_turbo/data/"):
            continue
        lower_name = name.lower()
        if Path(name).name.lower() in ROM_PAYLOAD_NAMES or lower_name.endswith(
            GAME_PAYLOAD_SUFFIXES,
        ):
            rom_payloads.append(name)
    checks = {
        "expected_filename": wheel.name in expected_names,
        "expected_python_tag": tag in PYTHON_TAGS,
        "version_in_filename": version in wheel.name,
        "expected_extension": expected_extension in names,
        "all_public_cores": all(
            f"env_stableretro_turbo/cores/{core}_libretro{core_suffix}" in names
            for core in PUBLIC_CORES
        ),
        "all_public_core_json": all(
            f"env_stableretro_turbo/cores/{core}.json" in names for core in PUBLIC_CORES
        ),
        "no_rom_payloads": not rom_payloads,
        "has_retro_vec_env_source": "env_stableretro_turbo/vec_env.py" in names,
    }
    with zipfile.ZipFile(wheel) as zf:
        init = zf.read("env_stableretro_turbo/__init__.py").decode("utf-8")
        vec_env = zf.read("env_stableretro_turbo/vec_env.py").decode("utf-8")
    checks.update(
        {
            "exports_retro_vec_env": "RetroVecEnv" in init,
            "does_not_export_retro_vector_env": "RetroVectorEnv" not in init,
            "class_retro_vec_env": "class RetroVecEnv" in vec_env,
            "does_not_define_retro_vector_env": "class RetroVectorEnv" not in vec_env,
            "uses_private_retro_vec_env_binding": "_RetroVecEnv" in vec_env,
            "does_not_use_public_retro_vec_env_binding": "_retro.RetroVecEnv"
            not in vec_env,
        },
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
    macos_wheels = args.macos_wheel or expected_macos_wheels(version)
    linux_wheels = args.linux_wheel or expected_linux_wheels(version)
    results = [
        *(audit_wheel(wheel, version, "macos-arm64") for wheel in macos_wheels),
        *(audit_wheel(wheel, version, "linux-x86_64") for wheel in linux_wheels),
    ]
    assert_audit_passed(results)
    print(json.dumps(results, indent=2))


def audit_sdist(sdist: Path, version: str) -> dict[str, object]:
    expected_name = expected_sdist(version).name
    compiled_artifacts: list[str] = []
    rom_payloads: list[str] = []
    unsafe_paths: list[str] = []
    source_core_dirs: set[str] = set()
    data_platforms: set[str] = set()
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
    for name in names:
        path = Path(name)
        parts = path.parts
        if path.is_absolute() or ".." in parts:
            unsafe_paths.append(name)
            continue
        rel = Path(*parts[1:]) if len(parts) > 1 else Path()
        if len(parts) >= 3 and parts[1] == "cores":
            core_dir = parts[2]
            if core_dir in KNOWN_CORE_SOURCE_DIRS:
                source_core_dirs.add(core_dir)
        if (
            len(parts) >= 5
            and parts[1:3] == ("env_stableretro_turbo", "data")
            and parts[3] in {"stable", "experimental", "contrib"}
            and "-" in parts[4]
        ):
            data_platforms.add(data_dir_platform(parts[4]))
        if rel.suffix in IGNORED_FILE_SUFFIXES:
            compiled_artifacts.append(name)
        if is_rom_payload(rel):
            rom_payloads.append(name)
    root = f"env_stableretro_turbo-{version}"
    public_platforms = frozenset(PUBLIC_DATA_PLATFORMS.split(","))
    checks = {
        "expected_filename": sdist.name == expected_name,
        "single_versioned_root": bool(names)
        and all(name == root or name.startswith(f"{root}/") for name in names),
        "safe_paths": not unsafe_paths,
        "no_compiled_artifacts": not compiled_artifacts,
        "no_rom_payloads": not rom_payloads,
        "all_public_core_sources": PUBLIC_CORE_SOURCE_DIRS <= source_core_dirs,
        "only_public_core_sources": source_core_dirs <= PUBLIC_CORE_SOURCE_DIRS,
        "only_public_data_platforms": data_platforms <= public_platforms,
        "has_bundled_saved_states": any(
            name.startswith(f"{root}/env_stableretro_turbo/data/") and name.endswith(".state")
            for name in names
        ),
        "no_libzip_regression_corpus": not any(
            name.startswith(f"{root}/third-party/libzip/regress/") for name in names
        ),
        "no_disabled_capnproto_sources": not any(
            name.startswith(f"{root}/third-party/capnproto/") for name in names
        ),
        "no_disabled_gtest_sources": not any(
            name.startswith(f"{root}/third-party/gtest/") for name in names
        ),
        "no_core_test_rom_corpus": not any(
            name.startswith(f"{root}/cores/gba/cinema/") for name in names
        ),
        "no_prebuilt_genesis_assets": not any(
            name.startswith(f"{root}/cores/genesis/builds/") for name in names
        ),
        "no_unused_pybind11_trees": not any(
            name.startswith(f"{root}/third-party/pybind11/{unused}/")
            for name in names
            for unused in ("docs", "pybind11", "tests", "tools")
        ),
        "within_pypi_file_limit": sdist.stat().st_size < 100_000_000,
        "has_setup_py": f"{root}/setup.py" in names,
        "has_version_file": f"{root}/env_stableretro_turbo/VERSION.txt" in names,
    }
    return {
        "sdist": str(sdist),
        "checks": checks,
        "compiled_artifacts": compiled_artifacts,
        "rom_payloads": rom_payloads,
        "unsafe_paths": unsafe_paths,
        "source_core_dirs": sorted(source_core_dirs),
        "data_platforms": sorted(data_platforms),
    }


def assert_sdist_audit_passed(result: dict[str, object]) -> None:
    checks = result["checks"]
    assert isinstance(checks, dict)
    failures = [key for key, value in checks.items() if not value]
    if failures:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(f"source distribution audit failed: {failures}")


def run(args_list: list[str], **kwargs: object) -> None:
    print("+", " ".join(shell_quote(arg) for arg in args_list))
    subprocess.run(args_list, check=True, **kwargs)


def smoke_wheel(args: argparse.Namespace) -> None:
    wheel = args.wheel.absolute()
    python = args.python.absolute()
    temp_root = release_temp_root()
    uv_env = os.environ.copy()
    uv_env.setdefault("UV_CACHE_DIR", str(temp_root / "uv-cache-stable-retro-build"))
    with tempfile.TemporaryDirectory(
        prefix="stable-retro-wheel-smoke.",
        dir=temp_root,
    ) as tmp:
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
            run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheel),
                ],
            )
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
                run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--target",
                        str(target),
                        str(wheel),
                    ],
                )

        code = """
from pathlib import Path

import numpy as np
import env_stableretro_turbo
from env_stableretro_turbo import _retro
print(env_stableretro_turbo.__file__)
print(_retro.__file__)
assert env_stableretro_turbo.__file__.startswith({target!r})
assert hasattr(_retro, "_RetroVecEnv")
assert not hasattr(_retro, "RetroVecEnv")
assert hasattr(env_stableretro_turbo, "RetroVecEnv")
assert not hasattr(env_stableretro_turbo, "RetroVectorEnv")

rom_path = Path({rom_path!r})
assert rom_path.is_file()
empty_info = Path({empty_info!r})
empty_info.write_text('{{"info": {{}}}}', encoding="utf-8")
env = env_stableretro_turbo.RetroVecEnv(
    "Dr88-FamiconIntro",
    state=env_stableretro_turbo.State.NONE,
    num_envs=2,
    num_threads=1,
    rom_path=str(rom_path),
    info=str(empty_info),
    scenario=str(empty_info),
    obs_copy="copy",
    obs_resize=(84, 84),
    obs_grayscale=True,
    obs_resize_algorithm="nearest",
    obs_layout="hwc",
    frame_skip=1,
    frame_stack=1,
    render_mode="rgb_array",
)
try:
    assert env.supports_live_snapshots is True
    env.reset(seed=17)
    warmup = np.zeros((2, env.num_buttons), dtype=np.uint8)
    env.step(warmup)
    handles = env.capture_snapshots(
        np.asarray([True, False], dtype=np.bool_)
    )
    assert handles[0] is not None
    assert handles[0].nbytes > 0
    assert handles[1] is None

    reset_options = {{
        "reset_mask": np.asarray([True, True], dtype=np.bool_),
        "state_indices": np.asarray([-1, -1], dtype=np.int32),
        "snapshots": [handles[0], handles[0]],
    }}
    restored, restored_infos = env.reset(options=reset_options)
    np.testing.assert_array_equal(restored[0], restored[1])
    assert restored_infos["start_source"].tolist() == [1, 1]

    replay_actions = np.zeros_like(warmup)
    replay_actions[:, 0] = 1
    first = tuple(
        np.asarray(value).copy() for value in env.step(replay_actions)[:4]
    )
    env.reset(options=reset_options)
    second = env.step(replay_actions)
    for expected, actual in zip(first, second[:4], strict=True):
        np.testing.assert_array_equal(expected, actual)
finally:
    env.close()
""".format(
            target=str(target),
            rom_path=str(REPO_ROOT / "tests" / "roms" / "Dr88-FamiconIntro.nes"),
            empty_info=str(target / "snapshot_smoke_info.json"),
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stable-retro")
        run([str(python), "-c", code], cwd=temp_root, env=env)


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
    wheels = [*expected_macos_wheels(version), *expected_linux_wheels(version)]
    sdist = expected_sdist(version)
    results = [
        *(
            audit_wheel(wheel, version, "macos-arm64")
            for wheel in expected_macos_wheels(version)
        ),
        *(
            audit_wheel(wheel, version, "linux-x86_64")
            for wheel in expected_linux_wheels(version)
        ),
    ]
    assert_audit_passed(results)
    sdist_result = audit_sdist(sdist, version)
    assert_sdist_audit_passed(sdist_result)
    distributions = [*wheels, sdist]
    run(
        [
            str(PYTHON),
            "-m",
            "twine",
            "check",
            *(str(distribution) for distribution in distributions),
        ],
    )
    hashes = {str(distribution): sha256(distribution) for distribution in distributions}
    print(
        json.dumps(
            {"wheel_audits": results, "sdist_audit": sdist_result, "sha256": hashes},
            indent=2,
        ),
    )
    print()
    print(publish_note(version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-version",
        help="Validate package name and version",
    )
    check.add_argument("--version")
    check.set_defaults(func=check_version)

    bump = subparsers.add_parser(
        "bump-version",
        help="Print or write the next post version",
    )
    bump.add_argument(
        "--to",
        help="Set an explicit target version instead of incrementing",
    )
    bump.add_argument(
        "--write",
        action="store_true",
        help="Write the target to env_stableretro_turbo/VERSION.txt",
    )
    bump.set_defaults(func=bump_version)

    pypi = subparsers.add_parser(
        "check-pypi",
        help="Check whether a PyPI version is still unused",
    )
    pypi.add_argument("--version")
    pypi.add_argument("--package")
    pypi.set_defaults(func=check_pypi)

    latest = subparsers.add_parser(
        "latest-pypi",
        help="Print the latest non-yanked PyPI version",
    )
    latest.add_argument(
        "--fail-if-mismatch",
        action="store_true",
        help="Fail if PyPI info.version differs from the computed latest non-yanked release",
    )
    latest.set_defaults(func=latest_pypi)

    prepare = subparsers.add_parser(
        "prepare-sources",
        help="Create clean macOS/Linux source copies",
    )
    prepare.add_argument("--version")
    prepare.add_argument("--root", type=Path)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=prepare_sources)

    commands = subparsers.add_parser(
        "build-commands",
        help="Print platform build commands",
    )
    commands.add_argument("--version")
    commands.add_argument("--macos-src")
    commands.add_argument("--linux-src")
    commands.set_defaults(func=build_commands)

    build = subparsers.add_parser(
        "build-platform",
        help="Build one release wheel platform",
    )
    build.add_argument("--platform", choices=RELEASE_PLATFORMS, required=True)
    build.add_argument("--version")
    build.set_defaults(func=build_platform)

    sdist = subparsers.add_parser(
        "build-sdist",
        help="Build and audit the ROM-free source distribution",
    )
    sdist.add_argument("--version")
    sdist.set_defaults(func=build_sdist)

    audit = subparsers.add_parser(
        "audit-wheels",
        help="Audit macOS and Linux wheel contents",
    )
    audit.add_argument("--version")
    audit.add_argument("--macos-wheel", type=Path, action="append")
    audit.add_argument("--linux-wheel", type=Path, action="append")
    audit.set_defaults(func=audit_wheels)

    smoke = subparsers.add_parser("smoke-wheel", help="Install and import-test a wheel")
    smoke.add_argument("wheel", type=Path)
    smoke.add_argument("--python", type=Path, default=PYTHON)
    smoke.add_argument("--installer", choices=("auto", "uv", "pip"), default="auto")
    smoke.set_defaults(func=smoke_wheel)

    smoke_macos = subparsers.add_parser(
        "smoke-macos-wheel",
        help="Install and import-test a macOS wheel",
    )
    smoke_macos.add_argument("wheel", type=Path)
    smoke_macos.add_argument("--python", type=Path, default=PYTHON)
    smoke_macos.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
    )
    smoke_macos.set_defaults(func=smoke_wheel)

    final = subparsers.add_parser(
        "final-check",
        help="Audit distributions, run twine check, hash, and print publishing handoff",
    )
    final.add_argument("--version")
    final.set_defaults(func=final_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import fnmatch
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_BENCH = "/root/stable-retro-bench"
REMOTE_CHECKOUT = "/root/stable-retro-turbo"
DEFAULT_GAME = "SuperMarioBros-Nes-v0"
DEFAULT_PROFILE = "supermario-level1-1"
UPSTREAM_STABLE_RETRO_GIT_URL = "https://github.com/Farama-Foundation/Stable-Retro.git"
UPSTREAM_STABLE_RETRO_GIT_REF = "main"
DEFAULT_ROM = (
    REPO_ROOT
    / "stable_retro"
    / "data"
    / "stable"
    / DEFAULT_GAME
    / "rom.nes"
)
CPU_REQUEST = 16.0
MEMORY_MB = 16384
PYTHON_VERSION = os.environ.get("MODAL_BENCHMARK_PYTHON_VERSION", "3.14")
NUMPY_VERSION = os.environ.get("MODAL_BENCHMARK_NUMPY_VERSION", "2.5.0")

CHECKOUT_ARCHIVE_IGNORE = [
    ".codex",
    ".codex/**",
    ".git",
    ".git/**",
    ".venv*",
    ".venv*/**",
    ".mypy_cache",
    ".mypy_cache/**",
    ".pytest_cache",
    ".pytest_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    ".uv-cache",
    ".uv-cache/**",
    "artifacts",
    "artifacts/**",
    "build",
    "build/**",
    "CMakeFiles",
    "**/CMakeFiles/**",
    "CMakeCache.txt",
    "CTestTestfile.cmake",
    "dist",
    "dist/**",
    "docs",
    "docs/**",
    "Makefile",
    "cmake_*.cmake",
    "install_manifest.txt",
    "_CPack_Packages",
    "CPack*Config.cmake",
    "wheelhouse*",
    "wheelhouse*/**",
    "__pycache__",
    "*.egg-info",
    "*.pyc",
    "*.o",
    "**/*.o",
    "*.o.tmp",
    "**/*.o.tmp",
    "*.obj",
    "**/*.obj",
    "*.a",
    "**/*.a",
    "*.tmp",
    "**/*.tmp",
    "*.so",
    "**/*.so",
    "*.dylib",
    "**/*.dylib",
    "*.egg-info",
    "*.egg-info/**",
    "architecture.png",
    "logo.png",
    "src/ui/logo.icns",
    "stable_retro/data/contrib/**",
    "stable_retro/data/experimental/**",
    "stable_retro/data/stable/*/rom.*",
    "tests",
    "tests/**",
    "third-party/libzip/regress/**",
]

app = modal.App("stable-retro-turbo-cpu-benchmarks")

image = (
    modal.Image.from_registry(f"python:{PYTHON_VERSION}-slim-bookworm")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "git",
        "libgl1-mesa-dev",
        "pkg-config",
        "zlib1g-dev",
        "libbz2-dev",
    )
    .pip_install(
        "setuptools==81.0.0",
        "wheel==0.45.1",
        f"numpy=={NUMPY_VERSION}",
        "torch==2.12.1",
        "stable-baselines3==2.9.0",
    )
    .add_local_file(
        REPO_ROOT / "scripts" / "benchmark_vec_env.py",
        f"{REMOTE_BENCH}/scripts/benchmark_vec_env.py",
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "scripts" / "benchmark_sb3_ppo.py",
        f"{REMOTE_BENCH}/scripts/benchmark_sb3_ppo.py",
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "scripts" / "benchmark_vec_env.json",
        f"{REMOTE_BENCH}/scripts/benchmark_vec_env.json",
        copy=True,
    )
    .workdir(REMOTE_BENCH)
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_text(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def normalize_package_source(package_source: str, package_version: str) -> str:
    source = package_source.strip().lower()
    if source not in {"checkout", "version", "upstream-git"}:
        raise ValueError(
            "--package-source must be 'checkout', 'version', or 'upstream-git'",
        )
    if source == "version" and not package_version.strip():
        raise ValueError("--package-version is required with --package-source=version")
    if source != "version" and package_version.strip():
        raise ValueError("--package-version requires --package-source=version")
    return source


def normalize_upstream_git_ref(package_source: str, upstream_git_ref: str) -> str:
    ref = upstream_git_ref.strip()
    if package_source == "upstream-git" and not ref:
        return UPSTREAM_STABLE_RETRO_GIT_REF
    if package_source != "upstream-git" and ref != UPSTREAM_STABLE_RETRO_GIT_REF:
        raise ValueError("--upstream-git-ref requires --package-source=upstream-git")
    return ref


def archive_path_ignored(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    parts = rel.split("/")
    ignored_dirs = {
        ".codex",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        "artifacts",
        "build",
        "CMakeFiles",
        "docs",
        "__pycache__",
        "_CPack_Packages",
        "tests",
    }
    if any(part in ignored_dirs or part.startswith(".venv") for part in parts):
        return True
    if parts[:3] == ["stable_retro", "data", "stable"]:
        if len(parts) < 4 or parts[3] != DEFAULT_GAME:
            return True
    if parts[:2] == ["stable_retro", "data"] and len(parts) >= 3:
        if parts[2] in {"contrib", "experimental"}:
            return True
    if parts[0] == "cores" and len(parts) >= 2:
        if len(parts) != 2 and parts[1] != "nes":
            return True
    for pattern in CHECKOUT_ARCHIVE_IGNORE:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def build_checkout_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(REPO_ROOT.rglob("*")):
            if archive_path_ignored(path) or path.is_dir():
                continue
            if not (path.is_file() or path.is_symlink()):
                continue
            tar.add(
                path,
                arcname=path.relative_to(REPO_ROOT).as_posix(),
                recursive=False,
            )
    return buffer.getvalue()


def stat_summary(values: list[float]) -> dict[str, float]:
    import statistics

    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_env_output(stdout: str) -> dict[str, Any]:
    config_line = next(
        (line for line in stdout.splitlines() if line.startswith("profile=")),
        "",
    )
    result_line = next(
        (line for line in stdout.splitlines() if " steps/s " in line),
        "",
    )
    match = re.search(
        r"^(?P<name>[^:]+): (?P<sps>[0-9.]+) steps/s "
        r"\((?P<steps>[0-9]+) steps in (?P<seconds>[0-9.]+)s\)",
        result_line,
    )
    if not match:
        raise RuntimeError(f"could not parse env benchmark output:\n{stdout}")
    return {
        "name": match.group("name"),
        "steps_per_second": float(match.group("sps")),
        "steps": int(match.group("steps")),
        "seconds": float(match.group("seconds")),
        "config_line": config_line,
        "stdout": stdout,
    }


def run_checked(
    command: list[str],
    *,
    cwd: str = REMOTE_BENCH,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "remote command failed\n"
            f"command={command!r}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return proc


@app.function(image=image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=3600)
def run_benchmark(
    rom_bytes: bytes,
    checkout_archive_bytes: bytes,
    *,
    package_source: str,
    package_version: str,
    upstream_git_ref: str,
    profile: str,
    env_backend: str,
    env_num_envs: int,
    repeats: int,
    env_seconds: float,
    env_warmup_steps: int,
    ppo_warmup_updates: int,
    ppo_measured_updates: int,
    ppo_n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    device: str,
    env_only: bool,
) -> dict[str, Any]:
    import os
    import platform
    import shutil
    import sys

    package_source = normalize_package_source(package_source, package_version)
    upstream_git_ref = normalize_upstream_git_ref(package_source, upstream_git_ref)
    scripts_dir = Path(REMOTE_BENCH) / "scripts"
    benchmark_cwd = "/tmp"
    upstream_git_commit = None
    if package_source == "version":
        run_checked(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                f"stable-retro-turbo=={package_version}",
            ],
        )
        sys.path[:] = [
            path
            for path in sys.path
            if Path(path or ".").resolve()
            not in {Path(REMOTE_BENCH), Path(REMOTE_CHECKOUT)}
        ]
    elif package_source == "upstream-git":
        spec = f"git+{UPSTREAM_STABLE_RETRO_GIT_URL}"
        if upstream_git_ref:
            spec += f"@{upstream_git_ref}"
            resolved_ref = run_checked(
                [
                    "git",
                    "ls-remote",
                    UPSTREAM_STABLE_RETRO_GIT_URL,
                    upstream_git_ref,
                ],
                cwd="/tmp",
            ).stdout.strip()
            upstream_git_commit = (
                resolved_ref.split()[0] if resolved_ref else upstream_git_ref
            )
        run_checked(
            [
                "python",
                "-m",
                "pip",
                "uninstall",
                "-y",
                "stable-retro-turbo",
                "stable-retro",
            ],
        )
        build_env = os.environ.copy()
        build_env.update(
            {
                "CMAKE_ARGS": (
                    "-DCMAKE_BUILD_TYPE=Release -DBUILD_CORES=nes "
                    "-DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DBUILD_N64=OFF"
                ),
                "STABLE_RETRO_PUBLIC_CORES": "fceumm",
                "STABLE_RETRO_PUBLIC_DATA_PLATFORMS": "Nes",
            },
        )
        run_checked(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-build-isolation",
                "--no-deps",
                spec,
            ],
            env=build_env,
        )
        sys.path[:] = [
            path
            for path in sys.path
            if Path(path or ".").resolve()
            not in {Path(REMOTE_BENCH), Path(REMOTE_CHECKOUT)}
        ]
    else:
        if not checkout_archive_bytes:
            raise ValueError("checkout archive bytes are required for checkout mode")
        checkout = Path(REMOTE_CHECKOUT)
        if checkout.exists():
            shutil.rmtree(checkout)
        checkout.mkdir(parents=True, exist_ok=True)
        archive_buffer = io.BytesIO(checkout_archive_bytes)
        with tarfile.open(fileobj=archive_buffer, mode="r:gz") as tar:
            tar.extractall(checkout, filter="data")
        run_checked(["python", "-m", "pip", "uninstall", "-y", "stable-retro-turbo"])
        build_env = os.environ.copy()
        build_env.update(
            {
                "CMAKE_ARGS": (
                    "-DCMAKE_BUILD_TYPE=Release -DBUILD_CORES=nes "
                    "-DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DBUILD_N64=OFF"
                ),
                "STABLE_RETRO_PUBLIC_CORES": "fceumm",
                "STABLE_RETRO_PUBLIC_DATA_PLATFORMS": "Nes",
            },
        )
        run_checked(
            ["python", "setup.py", "build_ext", "--inplace"],
            cwd=REMOTE_CHECKOUT,
            env=build_env,
        )
        run_checked(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "-e",
                ".",
            ],
            cwd=REMOTE_CHECKOUT,
        )
        scripts_dir = Path(REMOTE_CHECKOUT) / "scripts"
        benchmark_cwd = REMOTE_CHECKOUT

    import stable_retro

    package_rom = (
        Path(stable_retro.__file__).resolve().parent
        / "data"
        / "stable"
        / DEFAULT_GAME
        / "rom.nes"
    )
    package_rom.parent.mkdir(parents=True, exist_ok=True)
    package_rom.write_bytes(rom_bytes)

    modal_info = {
        "cpu_request": CPU_REQUEST,
        "memory_mb": MEMORY_MB,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "remote_benchmark_harness": REMOTE_BENCH,
        "remote_checkout": REMOTE_CHECKOUT if package_source == "checkout" else None,
        "remote_upstream_git_url": (
            UPSTREAM_STABLE_RETRO_GIT_URL
            if package_source == "upstream-git"
            else None
        ),
        "remote_upstream_git_ref": (
            upstream_git_ref if package_source == "upstream-git" else None
        ),
        "remote_upstream_git_commit": upstream_git_commit,
        "remote_rom_path": str(package_rom),
    }
    profiles_json = str(scripts_dir / "benchmark_vec_env.json")

    import_check = run_checked(
        [
            "python",
            "-c",
            (
                "import stable_retro, stable_retro._retro; "
                "import importlib.metadata as md; "
                "print(stable_retro.__file__); "
                "print(stable_retro._retro.__file__); "
                "print(str(getattr(stable_retro, '__version__', 'unknown')).strip()); "
                "\nfor name in ('stable-retro-turbo', 'stable-retro'):\n"
                "    try:\n"
                "        print(name)\n"
                "        print(md.version(name))\n"
                "        break\n"
                "    except md.PackageNotFoundError:\n"
                "        pass\n"
                "else:\n"
                "    print('unknown')\n"
                "    print('unknown')\n"
            ),
        ],
        cwd=benchmark_cwd,
    ).stdout.strip().splitlines()

    rom_check = run_checked(
        [
            "python",
            "-c",
            (
                "import stable_retro as retro; "
                f"print(retro.data.get_romfile_path({DEFAULT_GAME!r}))"
            ),
        ],
        cwd=benchmark_cwd,
    ).stdout.strip()

    env_runs = []
    for _ in range(repeats):
        command = [
            "python",
            str(scripts_dir / "benchmark_vec_env.py"),
            "--profiles-json",
            profiles_json,
            "--profile",
            profile,
            "--backend",
            env_backend,
            "--seconds",
            str(env_seconds),
            "--warmup-steps",
            str(env_warmup_steps),
        ]
        if env_num_envs > 0:
            command.extend(["--num-envs", str(env_num_envs)])
        proc = run_checked(
            command,
            cwd=benchmark_cwd,
        )
        env_runs.append(parse_env_output(proc.stdout))

    ppo_runs = []
    if not env_only:
        for idx in range(repeats):
            output_path = Path(f"/tmp/stable-retro-modal-ppo-{idx}.json")
            proc = run_checked(
                [
                    "python",
                    str(scripts_dir / "benchmark_sb3_ppo.py"),
                    "--package-source",
                    "checkout" if package_source == "checkout" else "installed",
                    "--profiles-json",
                    profiles_json,
                    "--profile",
                    profile,
                    "--warmup-updates",
                    str(ppo_warmup_updates),
                    "--measured-updates",
                    str(ppo_measured_updates),
                    "--n-steps",
                    str(ppo_n_steps),
                    "--batch-size",
                    str(ppo_batch_size),
                    "--n-epochs",
                    str(ppo_n_epochs),
                    "--device",
                    device,
                    "--json-output",
                    str(output_path),
                ],
                cwd=benchmark_cwd,
            )
            result = json.loads(output_path.read_text())
            result["stdout"] = proc.stdout
            ppo_runs.append(result)

    return {
        "kind": "stable-retro-turbo-modal-benchmark-v2",
        "target": {
            "package_source": package_source,
            "package_version": package_version if package_source == "version" else None,
            "upstream_git_url": (
                UPSTREAM_STABLE_RETRO_GIT_URL
                if package_source == "upstream-git"
                else None
            ),
            "upstream_git_ref": (
                upstream_git_ref if package_source == "upstream-git" else None
            ),
            "upstream_git_commit": upstream_git_commit,
        },
        "profile": profile,
        "modal": modal_info,
        "runtime": {
            "stable_retro_file": import_check[0] if len(import_check) > 0 else None,
            "stable_retro_extension": import_check[1] if len(import_check) > 1 else None,
            "stable_retro_source_version": import_check[2]
            if len(import_check) > 2
            else None,
            "stable_retro_distribution_name": import_check[3]
            if len(import_check) > 3
            else None,
            "stable_retro_distribution_version": import_check[4]
            if len(import_check) > 4
            else None,
            "stable_retro_turbo_distribution_version": (
                import_check[4]
                if len(import_check) > 4 and import_check[3] == "stable-retro-turbo"
                else None
            ),
            "rom_path": rom_check,
        },
        "env": {
            "runs": env_runs,
            "summary": stat_summary([run["steps_per_second"] for run in env_runs]),
        },
        "sb3_ppo": {
            "runs": ppo_runs,
            "summary": None
            if env_only
            else {
                "train_steps_per_second": stat_summary(
                    [
                        run["timing"]["train_steps_per_second"]
                        for run in ppo_runs
                    ],
                ),
                "rollout_steps_per_second": stat_summary(
                    [
                        run["timing"]["rollout_steps_per_second"]
                        for run in ppo_runs
                    ],
                ),
                "rollout_seconds": stat_summary(
                    [run["timing"]["rollout_seconds"] for run in ppo_runs],
                ),
                "update_seconds": stat_summary(
                    [run["timing"]["update_seconds"] for run in ppo_runs],
                ),
            },
        },
    }


@app.local_entrypoint()
def main(
    output_json: str = "",
    package_source: str = "checkout",
    package_version: str = "",
    upstream_git_ref: str = UPSTREAM_STABLE_RETRO_GIT_REF,
    profile: str = DEFAULT_PROFILE,
    env_backend: str = "auto",
    env_num_envs: int = 0,
    repeats: int = 3,
    env_seconds: float = 30.0,
    env_warmup_steps: int = 32,
    ppo_warmup_updates: int = 1,
    ppo_measured_updates: int = 10,
    ppo_n_steps: int = 512,
    ppo_batch_size: int = 512,
    ppo_n_epochs: int = 4,
    device: str = "cpu",
    smoke: bool = False,
    env_only: bool = False,
) -> None:
    package_source = normalize_package_source(package_source, package_version)
    upstream_git_ref = normalize_upstream_git_ref(package_source, upstream_git_ref)
    if smoke:
        repeats = 1
        env_seconds = 2.0
        env_warmup_steps = 8
        ppo_warmup_updates = 0
        ppo_measured_updates = 1
        ppo_n_steps = 8
        ppo_batch_size = 16
        ppo_n_epochs = 1

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if env_backend not in {"auto", "native", "subproc", "dummy", "async"}:
        raise ValueError("--env-backend must be auto, native, subproc, dummy, or async")
    if env_num_envs < 0:
        raise ValueError("--env-num-envs must be non-negative")
    if not DEFAULT_ROM.exists():
        raise FileNotFoundError(f"ROM not found: {DEFAULT_ROM}")

    rom_bytes = DEFAULT_ROM.read_bytes()
    checkout_archive_bytes = (
        build_checkout_archive() if package_source == "checkout" else b""
    )
    result = run_benchmark.remote(
        rom_bytes,
        checkout_archive_bytes,
        package_source=package_source,
        package_version=package_version,
        upstream_git_ref=upstream_git_ref,
        profile=profile,
        env_backend=env_backend,
        env_num_envs=env_num_envs,
        repeats=repeats,
        env_seconds=env_seconds,
        env_warmup_steps=env_warmup_steps,
        ppo_warmup_updates=ppo_warmup_updates,
        ppo_measured_updates=ppo_measured_updates,
        ppo_n_steps=ppo_n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        device=device,
        env_only=env_only,
    )
    result["local"] = {
        "repo_root": str(REPO_ROOT),
        "git": {
            "commit": git_text("rev-parse", "HEAD"),
            "branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
            "status_short": git_text("status", "--short"),
        },
        "rom": {
            "path": str(DEFAULT_ROM),
            "bytes": len(rom_bytes),
            "sha256": sha256(rom_bytes),
        },
        "config": {
            "repeats": repeats,
            "env_seconds": env_seconds,
            "env_warmup_steps": env_warmup_steps,
            "ppo_warmup_updates": ppo_warmup_updates,
            "ppo_measured_updates": ppo_measured_updates,
            "ppo_n_steps": ppo_n_steps,
            "ppo_batch_size": ppo_batch_size,
            "ppo_n_epochs": ppo_n_epochs,
            "device": device,
            "smoke": smoke,
            "package_source": package_source,
            "package_version": package_version if package_source == "version" else None,
            "upstream_git_url": (
                UPSTREAM_STABLE_RETRO_GIT_URL
                if package_source == "upstream-git"
                else None
            ),
            "upstream_git_ref": (
                upstream_git_ref if package_source == "upstream-git" else None
            ),
            "env_backend": env_backend,
            "env_num_envs": env_num_envs if env_num_envs > 0 else None,
            "checkout_archive_bytes": len(checkout_archive_bytes),
            "checkout_archive_uploaded": package_source == "checkout",
        },
    }

    output_path = Path(output_json).expanduser() if output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    env_summary = result["env"]["summary"]
    print(
        "target="
        f"package_source={result['target']['package_source']} "
        f"package_version={result['target']['package_version']}"
    )
    print(
        "modal="
        f"cpu_request={result['modal']['cpu_request']} "
        f"memory_mb={result['modal']['memory_mb']} "
        f"os_cpu_count={result['modal']['os_cpu_count']} "
        f"affinity_cpu_count={result['modal']['affinity_cpu_count']}"
    )
    print(
        "env_steps_per_second="
        f"mean={env_summary['mean']:.1f} "
        f"stdev={env_summary['stdev']:.1f} "
        f"best={env_summary['max']:.1f} "
        f"runs={[round(run['steps_per_second'], 1) for run in result['env']['runs']]}"
    )
    if not env_only:
        train_summary = result["sb3_ppo"]["summary"]["train_steps_per_second"]
        print(
            "train_steps_per_second="
            f"mean={train_summary['mean']:.1f} "
            f"stdev={train_summary['stdev']:.1f} "
            f"best={train_summary['max']:.1f} "
            f"runs={[round(run['timing']['train_steps_per_second'], 1) for run in result['sb3_ppo']['runs']]}"
        )
    if output_path is not None:
        print(f"wrote_json={output_path}")

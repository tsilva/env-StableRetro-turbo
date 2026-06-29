from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/stable-retro-turbo"
DEFAULT_GAME = "SuperMarioBros-Nes-v0"
DEFAULT_PROFILE = "supermario-level1-1"
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
PYTHON_VERSION = "3.14"

IMAGE_IGNORE = [
    ".git",
    ".venv*",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    "artifacts",
    "build",
    "dist",
    "wheelhouse*",
    "__pycache__",
    "*.egg-info",
    "*.pyc",
    "*.so",
    "*.dylib",
]

app = modal.App("stable-retro-turbo-cpu-benchmarks")

image = (
    modal.Image.from_registry(f"python:{PYTHON_VERSION}-slim-bookworm")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "git",
        "pkg-config",
        "zlib1g-dev",
        "libbz2-dev",
    )
    .pip_install(
        "setuptools==81.0.0",
        "wheel==0.45.1",
        "numpy==2.5.0",
        "torch==2.12.1",
        "stable-baselines3==2.9.0",
    )
    .add_local_dir(REPO_ROOT, REMOTE_REPO, copy=True, ignore=IMAGE_IGNORE)
    .workdir(REMOTE_REPO)
    .run_commands(
        (
            "CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DBUILD_CORES=nes "
            "-DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DBUILD_N64=OFF' "
            "STABLE_RETRO_PUBLIC_CORES=fceumm "
            "STABLE_RETRO_PUBLIC_DATA_PLATFORMS=Nes "
            "python setup.py build_ext --inplace"
        ),
        "python -m pip install --no-build-isolation -e .",
    )
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
    if source not in {"checkout", "version"}:
        raise ValueError("--package-source must be 'checkout' or 'version'")
    if source == "version" and not package_version.strip():
        raise ValueError("--package-version is required with --package-source=version")
    if source == "checkout" and package_version.strip():
        raise ValueError("--package-version requires --package-source=version")
    return source


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
    cwd: str = REMOTE_REPO,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
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
    *,
    package_source: str,
    package_version: str,
    profile: str,
    repeats: int,
    env_seconds: float,
    env_warmup_steps: int,
    ppo_warmup_updates: int,
    ppo_measured_updates: int,
    ppo_n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    device: str,
) -> dict[str, Any]:
    import os
    import platform
    import sys

    package_source = normalize_package_source(package_source, package_version)
    if package_source == "version":
        run_checked(["python", "-m", "pip", "uninstall", "-y", "stable-retro-turbo"])
        run_checked(
            [
                "python",
                "-m",
                "pip",
                "install",
                f"stable-retro-turbo=={package_version}",
            ],
        )
        sys.path[:] = [
            path
            for path in sys.path
            if Path(path or ".").resolve() != Path(REMOTE_REPO)
        ]

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
        "remote_repo": REMOTE_REPO,
        "remote_rom_path": str(package_rom),
    }
    benchmark_cwd = REMOTE_REPO if package_source == "checkout" else "/tmp"
    scripts_dir = Path(REMOTE_REPO) / "scripts"
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
                "print(md.version('stable-retro-turbo'))"
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
        proc = run_checked(
            [
                "python",
                str(scripts_dir / "benchmark_vec_env.py"),
                "--profiles-json",
                profiles_json,
                "--profile",
                profile,
                "--seconds",
                str(env_seconds),
                "--warmup-steps",
                str(env_warmup_steps),
            ],
            cwd=benchmark_cwd,
        )
        env_runs.append(parse_env_output(proc.stdout))

    ppo_runs = []
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
        "kind": "stable-retro-turbo-modal-benchmark-v1",
        "target": {
            "package_source": package_source,
            "package_version": package_version if package_source == "version" else None,
        },
        "profile": profile,
        "modal": modal_info,
        "runtime": {
            "stable_retro_file": import_check[0] if len(import_check) > 0 else None,
            "stable_retro_extension": import_check[1] if len(import_check) > 1 else None,
            "stable_retro_turbo_distribution_version": import_check[2]
            if len(import_check) > 2
            else None,
            "rom_path": rom_check,
        },
        "env": {
            "runs": env_runs,
            "summary": stat_summary([run["steps_per_second"] for run in env_runs]),
        },
        "sb3_ppo": {
            "runs": ppo_runs,
            "summary": {
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
    profile: str = DEFAULT_PROFILE,
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
) -> None:
    package_source = normalize_package_source(package_source, package_version)
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
    if not DEFAULT_ROM.exists():
        raise FileNotFoundError(f"ROM not found: {DEFAULT_ROM}")

    rom_bytes = DEFAULT_ROM.read_bytes()
    result = run_benchmark.remote(
        rom_bytes,
        package_source=package_source,
        package_version=package_version,
        profile=profile,
        repeats=repeats,
        env_seconds=env_seconds,
        env_warmup_steps=env_warmup_steps,
        ppo_warmup_updates=ppo_warmup_updates,
        ppo_measured_updates=ppo_measured_updates,
        ppo_n_steps=ppo_n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        device=device,
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
        },
    }

    output_path = Path(output_json).expanduser() if output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    env_summary = result["env"]["summary"]
    train_summary = result["sb3_ppo"]["summary"]["train_steps_per_second"]
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
    print(
        "train_steps_per_second="
        f"mean={train_summary['mean']:.1f} "
        f"stdev={train_summary['stdev']:.1f} "
        f"best={train_summary['max']:.1f} "
        f"runs={[round(run['timing']['train_steps_per_second'], 1) for run in result['sb3_ppo']['runs']]}"
    )
    if output_path is not None:
        print(f"wrote_json={output_path}")

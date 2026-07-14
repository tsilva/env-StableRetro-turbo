"""Run the Atari RetroVecEnv benchmark on isolated Modal CPU."""

from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

import modal

from modal_benchmark import (
    CPU_REQUEST,
    MEMORY_MB,
    REMOTE_CHECKOUT,
    REPO_ROOT,
    archive_path_ignored,
    git_text,
    image,
)


GAME = "Breakout-Atari2600-v0"
LOCAL_GAME_DIR = Path.home() / "roms" / "stable_retro" / "data" / "stable" / GAME
REMOTE_ARTIFACT_TEMPLATE = "/tmp/atari-retrovec-{info_filter}.json"

app = modal.App("stable-retro-turbo-atari-cpu-benchmark")
atari_image = image.add_local_file(
    REPO_ROOT / "scripts" / "modal_benchmark.py",
    "/root/modal_benchmark.py",
    copy=True,
)


def build_checkout_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(REPO_ROOT.rglob("*")):
            rel = path.relative_to(REPO_ROOT)
            in_atari_core = rel.parts[:2] == ("cores", "atari2600")
            required_luajit_makefile = (
                rel.parts[:2] == ("third-party", "luajit")
                and path.name == "Makefile"
            )
            if path.is_dir() or (
                archive_path_ignored(path)
                and not in_atari_core
                and not required_luajit_makefile
            ):
                continue
            if in_atari_core and path.suffix in {".a", ".dylib", ".o", ".so"}:
                continue
            if not (path.is_file() or path.is_symlink()):
                continue
            tar.add(path, arcname=rel.as_posix(), recursive=False)
    return buffer.getvalue()


def _run(command: list[str], *, cwd: str, env=None) -> str:
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
            f"remote command failed: {command!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
    return proc.stdout


@app.function(image=atari_image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=3600)
def run_atari_benchmark(checkout_archive, integration_files, repeats, seconds, info_filters):
    checkout = Path(REMOTE_CHECKOUT)
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(checkout_archive), mode="r:gz") as tar:
        tar.extractall(checkout, filter="data")

    game_dir = checkout / "stable_retro" / "data" / "stable" / GAME
    game_dir.mkdir(parents=True, exist_ok=True)
    (checkout / "stable_retro" / "data" / "experimental").mkdir(parents=True)
    (checkout / "stable_retro" / "data" / "contrib").mkdir(parents=True)
    for name, payload in integration_files.items():
        (game_dir / name).write_bytes(payload)

    build_env = os.environ.copy()
    build_env.update(
        {
            "CMAKE_ARGS": (
                "-DCMAKE_BUILD_TYPE=Release -DBUILD_CORES=atari2600 "
                "-DBUILD_TESTS=OFF -DENABLE_CAPNPROTO=OFF -DBUILD_N64=OFF"
            ),
            "STABLE_RETRO_PUBLIC_CORES": "stella",
            "STABLE_RETRO_PUBLIC_DATA_PLATFORMS": "Atari2600",
        },
    )
    _run(["python", "setup.py", "build_ext", "--inplace"], cwd=REMOTE_CHECKOUT, env=build_env)
    _run(
        ["python", "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd=REMOTE_CHECKOUT,
        env=build_env,
    )
    results = {}
    stdout = {}
    for info_filter in info_filters:
        artifact = REMOTE_ARTIFACT_TEMPLATE.format(info_filter=info_filter)
        stdout[info_filter] = _run(
            [
                "python",
                "scripts/benchmark_atari.py",
                "--repeats",
                str(repeats),
                "--seconds",
                str(seconds),
                "--info-filter",
                info_filter,
                "--output-json",
                artifact,
            ],
            cwd=REMOTE_CHECKOUT,
        )
        results[info_filter] = json.loads(Path(artifact).read_text())
    return {
        "results": results,
        "cpu_request": CPU_REQUEST,
        "memory_mb": MEMORY_MB,
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "platform": platform.platform(),
        "stdout": stdout,
    }


@app.local_entrypoint()
def main(
    output_json: str = "artifacts/benchmarks/modal-atari-retrovec.json",
    repeats: int = 3,
    seconds: float = 30.0,
    info_filter: str = "none",
) -> None:
    info_filters = [value.strip() for value in info_filter.split(",") if value.strip()]
    if not info_filters or any(value not in {"all", "terminal", "none"} for value in info_filters):
        raise ValueError("info_filter must be a comma-separated list of all, terminal, or none")
    if len(set(info_filters)) != len(info_filters):
        raise ValueError("info_filter values must be unique")
    required = ("rom.a26", "rom.sha", "Start.state", "metadata.json", "data.json", "scenario.json")
    missing = [name for name in required if not (LOCAL_GAME_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Breakout integration files: {missing}")
    integration_files = {name: (LOCAL_GAME_DIR / name).read_bytes() for name in required}
    remote_result = run_atari_benchmark.remote(
        build_checkout_archive(),
        integration_files,
        repeats,
        seconds,
        info_filters,
    )
    result = {
        "results": remote_result.pop("results"),
        "modal": remote_result,
        "local": {
            "repo_root": str(REPO_ROOT),
            "branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git_text("rev-parse", "HEAD"),
            "status_short": git_text("status", "--short"),
        },
    }
    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for filter_name, filter_result in result["results"].items():
        summary = filter_result["result"]["summary"]
        samples = [round(run["steps_per_second"], 1) for run in filter_result["result"]["runs"]]
        print(
            f"retro_vec_stella[{filter_name}]: samples={samples} mean={summary['mean']:.1f} "
            f"stdev={summary['stdev']:.1f} best={summary['max']:.1f}",
        )
    print(f"wrote_json={output_path}")


if __name__ == "__main__":
    raise SystemExit("Run with `modal run scripts/modal_atari_benchmark.py`")

from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

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
REMOTE_ARTIFACT = "/tmp/atari-stella-vs-alepy.json"

app = modal.App("stable-retro-turbo-atari-cpu-benchmark")
atari_image = image.add_local_file(
    REPO_ROOT / "scripts" / "modal_benchmark.py",
    "/root/modal_benchmark.py",
    copy=True,
)


def build_atari_checkout_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(REPO_ROOT.rglob("*")):
            rel = path.relative_to(REPO_ROOT)
            in_atari_core = rel.parts[:2] == ("cores", "atari2600")
            required_luajit_makefile = (
                rel.parts[:2] == ("third-party", "luajit")
                and path.name == "Makefile"
            )
            generated_core_file = path.suffix in {".a", ".dylib", ".o", ".so"}
            if path.is_dir() or (in_atari_core and generated_core_file):
                continue
            if (
                not in_atari_core
                and not required_luajit_makefile
                and archive_path_ignored(path)
            ):
                continue
            if not (path.is_file() or path.is_symlink()):
                continue
            tar.add(path, arcname=rel.as_posix(), recursive=False)
    return buffer.getvalue()


def _run(command: list[str], *, cwd: str, env: dict[str, str] | None = None) -> str:
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
            f"remote command failed: {command!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
    return proc.stdout


@app.function(image=atari_image, cpu=CPU_REQUEST, memory=MEMORY_MB, timeout=3600)
def run_atari_benchmark(
    checkout_archive: bytes,
    integration_files: dict[str, bytes],
    *,
    repeats: int,
    seconds: float,
    upstream_stella_ref: str,
) -> dict[str, Any]:
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

    upstream_stella_commit = None
    if upstream_stella_ref:
        upstream_dir = Path("/tmp/stella-libretro")
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                upstream_stella_ref,
                "https://github.com/libretro/stella-libretro.git",
                str(upstream_dir),
            ],
            cwd="/tmp",
        )
        upstream_stella_commit = _run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(upstream_dir),
        ).strip()
        core_dir = checkout / "cores" / "atari2600"
        shutil.rmtree(core_dir)
        shutil.copytree(upstream_dir, core_dir, ignore=shutil.ignore_patterns(".git"))
        makefile_path = core_dir / "Makefile"
        makefile_path.write_text(
            makefile_path.read_text().replace("TARGET_NAME := stella2014", "TARGET_NAME := stella"),
        )
        libretro_path = core_dir / "libretro.cxx"
        source = libretro_path.read_text()
        start = source.rindex("void retro_run(void)\n{")
        source = source[:start] + r'''static bool stable_retro_audio_enabled = true;

static void stable_retro_run_internal(bool skip_render)
{
   static int16_t sampleBuffer[2048];
   uint32_t tiaSamplesPerFrame =
         (31400u * console->getFramerateDen()) /
         (console->getFramerateNum() ? console->getFramerateNum() : 1498u);
   if (tiaSamplesPerFrame > 1024)
      tiaSamplesPerFrame = 1024;

   bool updated = false;
   if (environ_cb(RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE, &updated) && updated)
      check_variables(false);

   update_input();
   TIA& tia = console->tia();
   tia.update();
   videoWidth = tia.width();
   videoHeight = tia.height();
   if (videoHeight > FRAME_BUFFER_MAX_LINES)
      videoHeight = FRAME_BUFFER_MAX_LINES;

   if (!skip_render)
   {
      if (framePixelBytes == 2)
         blend_frames_16(tia.currentFrameBuffer(), videoWidth, videoHeight);
      else
         blend_frames_32(tia.currentFrameBuffer(), videoWidth, videoHeight);
      video_cb(frameBuffer, videoWidth, videoHeight, videoWidth * framePixelBytes);
   }

   if (stable_retro_audio_enabled)
   {
      osystem.sound().processFragment(sampleBuffer, tiaSamplesPerFrame);
      apply_dc_block_filter(sampleBuffer, tiaSamplesPerFrame);
      if (low_pass_enabled)
         apply_low_pass_filter(sampleBuffer, tiaSamplesPerFrame);
      audio_batch_cb(sampleBuffer, tiaSamplesPerFrame);
   }
}

void retro_run(void)
{
   stable_retro_run_internal(false);
}

extern "C" RETRO_API void stable_retro_run_skip_render(void)
{
   stable_retro_run_internal(true);
}

extern "C" RETRO_API void stable_retro_set_audio_enabled(bool enabled)
{
   stable_retro_audio_enabled = enabled;
}
'''
        libretro_path.write_text(source)
        link_path = core_dir / "link.T"
        link_path.write_text(link_path.read_text().replace("global: retro_*;", "global: retro_*; stable_retro_*;"))

    _run(
        ["python", "-m", "pip", "install", "--no-deps", "ale-py==0.12.0"],
        cwd=REMOTE_CHECKOUT,
    )
    _run(
        ["python", "-m", "pip", "uninstall", "-y", "stable-retro-turbo"],
        cwd=REMOTE_CHECKOUT,
    )
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
    _run(
        ["python", "setup.py", "build_ext", "--inplace"],
        cwd=REMOTE_CHECKOUT,
        env=build_env,
    )
    _run(
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

    benchmark_command = [
        "python",
        "scripts/benchmark_atari_alepy.py",
        "--seconds",
        str(seconds),
        "--repeats",
        str(repeats),
        "--output-json",
        REMOTE_ARTIFACT,
    ]
    if upstream_stella_ref:
        # Stella save states are core-version-specific. A cold reset keeps the
        # workload comparable when evaluating a newer upstream core.
        benchmark_command.extend(["--state", "none"])
    stdout = _run(
        benchmark_command,
        cwd=REMOTE_CHECKOUT,
    )
    result = json.loads(Path(REMOTE_ARTIFACT).read_text())
    result["remote"] = {
        "cpu_request": CPU_REQUEST,
        "memory_mb": MEMORY_MB,
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "platform": platform.platform(),
        "remote_checkout": REMOTE_CHECKOUT,
        "upstream_stella_ref": upstream_stella_ref or None,
        "upstream_stella_commit": upstream_stella_commit,
        "stdout": stdout,
    }
    return result


@app.local_entrypoint()
def main(
    output_json: str = "artifacts/benchmarks/modal-atari-stella-vs-alepy.json",
    repeats: int = 3,
    seconds: float = 30.0,
    upstream_stella_ref: str = "",
) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    required = ("rom.a26", "rom.sha", "Start.state", "metadata.json", "data.json", "scenario.json")
    missing = [name for name in required if not (LOCAL_GAME_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Breakout integration files: {missing}")

    integration_files = {
        name: (LOCAL_GAME_DIR / name).read_bytes()
        for name in required
    }
    result = run_atari_benchmark.remote(
        build_atari_checkout_archive(),
        integration_files,
        repeats=repeats,
        seconds=seconds,
        upstream_stella_ref=upstream_stella_ref,
    )
    result["local"] = {
        "repo_root": str(REPO_ROOT),
        "branch": git_text("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_text("rev-parse", "HEAD"),
        "status_short": git_text("status", "--short"),
        "game_dir": str(LOCAL_GAME_DIR),
        "upstream_stella_ref": upstream_stella_ref or None,
    }

    output_path = Path(output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for condition in result["results"]:
        summary = condition["summary"]
        samples = [round(run["steps_per_second"], 1) for run in condition["runs"]]
        print(
            f"{condition['name']}: samples={samples} mean={summary['mean']:.1f} "
            f"stdev={summary['stdev']:.1f} best={summary['max']:.1f}",
        )
    print(f"wrote_json={output_path}")


if __name__ == "__main__":
    raise SystemExit("Run with `modal run scripts/modal_atari_benchmark.py`")

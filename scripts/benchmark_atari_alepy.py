"""Compare stable-retro Atari vector throughput with ale-py AtariVectorEnv."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_vec_env import _load_profiles
from stable_retro.atari_vec_env import ale_game_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = "atari-breakout"


def _default_output_path() -> Path:
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    return ROOT / "artifacts" / "benchmarks" / f"atari-alepy-{stamp}.json"


def _ale_rom_id_from_retro_game(game: str) -> str:
    return ale_game_id(game)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _parse_resize(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise SystemExit(f"Invalid resize value {value!r}; expected HEIGHTxWIDTH") from exc
    if height <= 0 or width <= 0:
        raise SystemExit(f"Invalid resize value {value!r}; dimensions must be positive")
    return height, width


def _run_alepy_child(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import ale_py
    from ale_py import AtariVectorEnv

    env = AtariVectorEnv(
        config["ale_game"],
        num_envs=config["num_envs"],
        num_threads=config["num_threads"],
        repeat_action_probability=0.0,
        img_height=config["resize"][0],
        img_width=config["resize"][1],
        grayscale=config["grayscale"],
        stack_num=config["frame_stack"],
        frameskip=config["frame_skip"],
        maxpool=config["maxpool_last_two"],
        noop_max=0,
        episodic_life=False,
        life_loss_info=False,
        reward_clipping=False,
        use_fire_reset=False,
    )
    obs, _info = env.reset(seed=0)
    action = np.zeros((config["num_envs"],), dtype=np.int64)
    for _ in range(config["warmup_steps"]):
        env.step(action)

    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < config["seconds"]:
        env.step(action)
        steps += config["num_envs"]
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "steps": steps,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "obs_shape": list(obs.shape),
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "ale_py_file": ale_py.__file__,
        "ale_py_version": getattr(ale_py, "__version__", None),
    }


def _run_v1_child(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import stable_retro as retro

    env = retro.AtariVecEnv(
        config["game"],
        state=retro.State.NONE,
        num_envs=config["num_envs"],
        num_threads=config["num_threads"],
        obs_resize=tuple(config["resize"]),
        obs_grayscale=config["grayscale"],
        frame_skip=config["frame_skip"],
        frame_stack=config["frame_stack"],
        maxpool_last_two=config["maxpool_last_two"],
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        use_fire_reset=False,
    )
    obs, _info = env.reset(seed=0)
    action = np.zeros((config["num_envs"],), dtype=np.int64)
    for _ in range(config["warmup_steps"]):
        env.step(action)

    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < config["seconds"]:
        env.step(action)
        steps += config["num_envs"]
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "steps": steps,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "obs_shape": list(obs.shape),
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "stable_retro_file": retro.__file__,
        "stable_retro_version": getattr(retro, "__version__", None),
        "backend": "atari-v1",
    }


def _run_child(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--child-backend",
        choices=("v1", "alepy"),
        required=True,
    )
    parser.add_argument("--child-config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.child_config)
    if args.child_backend == "v1":
        payload = _run_v1_child(config)
    else:
        payload = _run_alepy_child(config)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _run_condition(
    *,
    name: str,
    backend: str,
    config: dict[str, Any],
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    runs = []
    for repeat in range(1, repeats + 1):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        if config.get("stable_indexed_video"):
            env["STABLE_RETRO_ENABLE_ATARI_INDEXED_VIDEO"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-backend",
                backend,
                "--child-config",
                json.dumps(config, sort_keys=True),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{name} repeat {repeat} failed with {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        payload["repeat"] = repeat
        runs.append(payload)
        print(f"{name} repeat {repeat}: {payload['steps_per_second']:.1f} steps/s")
    values = [run["steps_per_second"] for run in runs]
    return {
        "name": name,
        "backend": backend,
        "runs": runs,
        "summary": _summary(values),
    }


def _build_config(args, profile) -> dict[str, Any]:
    resize = _parse_resize(profile.resize if args.resize is None else args.resize)
    num_envs = profile.num_envs if args.num_envs is None else args.num_envs
    num_threads = profile.num_threads if args.num_threads is None else args.num_threads
    ale_game = args.ale_game or _ale_rom_id_from_retro_game(profile.game)
    return {
        "game": profile.game,
        "ale_game": ale_game,
        "resize": list(resize),
        "grayscale": profile.grayscale,
        "frame_skip": profile.frame_skip if args.frame_skip is None else args.frame_skip,
        "frame_stack": profile.frame_stack
        if args.frame_stack is None
        else args.frame_stack,
        "resize_algorithm": profile.resize_algorithm,
        "maxpool_last_two": profile.maxpool_last_two and not args.no_maxpool_last_two,
        "num_envs": num_envs,
        "num_threads": num_threads,
        "obs_copy": args.obs_copy,
        "warmup_steps": args.warmup_steps,
        "seconds": args.seconds,
        "stable_indexed_video": args.stable_indexed_video,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--child-backend" in argv:
        return _run_child(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-json",
        default=str(Path(__file__).with_name("benchmark_atari_alepy.json")),
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--ale-game", default=None)
    parser.add_argument("--resize", default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument(
        "--obs-copy",
        choices=("copy", "safe_view", "unsafe_view"),
        default="safe_view",
    )
    parser.add_argument("--no-maxpool-last-two", action="store_true")
    parser.add_argument("--stable-indexed-video", action="store_true")
    parser.add_argument("--child-timeout", type=float, default=90.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")

    profiles = _load_profiles(Path(args.profiles_json))
    try:
        profile = profiles[args.profile]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown profile {args.profile!r}. Available: {available}") from exc

    config = _build_config(args, profile)
    output_json = args.output_json or _default_output_path()
    workload = {
        "profile": args.profile,
        "action_policy": "fixed_noop",
        "ale_game": config["ale_game"],
        **{
            key: config[key]
            for key in (
                "game",
                "resize",
                "grayscale",
                "frame_skip",
                "frame_stack",
                "maxpool_last_two",
                "num_envs",
                "num_threads",
                "warmup_steps",
                "seconds",
                "obs_copy",
            )
        },
    }
    print(
        "profile={profile} game={game} ale_game={ale_game} "
        "envs={num_envs} threads={num_threads} seconds={seconds} repeats={repeats}".format(
            profile=args.profile,
            game=config["game"],
            ale_game=config["ale_game"],
            num_envs=config["num_envs"],
            num_threads=config["num_threads"],
            seconds=args.seconds,
            repeats=args.repeats,
        ),
    )
    if args.dry_run:
        print(f"output_json={output_json}")
        print(json.dumps(workload, sort_keys=True))
        return 0

    results = [
        _run_condition(
            name="stable_retro_atari_v1",
            backend="v1",
            config=config,
            repeats=args.repeats,
            timeout=args.child_timeout,
        ),
        _run_condition(
            name="alepy",
            backend="alepy",
            config=config,
            repeats=args.repeats,
            timeout=args.child_timeout,
        ),
    ]
    artifact = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "platform": platform.platform(),
        "workload": workload,
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"artifact={output_json}")
    for result in results:
        samples = [round(run["steps_per_second"], 1) for run in result["runs"]]
        summary = result["summary"]
        print(
            f"{result['name']}: samples={samples} "
            f"mean={summary['mean']:.1f} stdev={summary['stdev']:.1f} "
            f"best={summary['max']:.1f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark Atari RetroVecEnv throughput with manual reset."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _run(args) -> dict:
    import stable_retro as retro
    from stable_retro.vec_env import RetroVecEnv

    runs = []
    for repeat in range(1, args.repeats + 1):
        env = RetroVecEnv(
            args.game,
            state=args.state,
            num_envs=args.num_envs,
            num_threads=args.num_threads,
            obs_copy=args.obs_copy,
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_grayscale=True,
            obs_resize_algorithm="area",
            obs_layout="chw",
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
            info_filter=args.info_filter,
        )
        action = np.zeros((args.num_envs, env.num_buttons), dtype=np.uint8)

        def step_and_reset():
            _obs, _rewards, terminated, truncated, _infos = env.step(action)
            done = terminated | truncated
            if done.any():
                env.reset(options={"reset_mask": done})
            return int(done.sum())

        try:
            env.reset(seed=0)
            for _ in range(args.warmup_steps):
                step_and_reset()

            steps = 0
            terminal_lanes = 0
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                terminal_lanes += step_and_reset()
                steps += args.num_envs
            elapsed = time.perf_counter() - start
            rate = steps / elapsed
        finally:
            env.close()
        runs.append(
            {
                "repeat": repeat,
                "steps": steps,
                "seconds": elapsed,
                "steps_per_second": rate,
                "terminal_lanes": terminal_lanes,
            },
        )
        print(f"repeat {repeat}: {rate:.1f} steps/s", flush=True)
    values = [run["steps_per_second"] for run in runs]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": {
            "game": args.game,
            "state": args.state,
            "num_envs": args.num_envs,
            "num_threads": args.num_threads,
            "resize": [84, 84],
            "grayscale": True,
            "frame_skip": 4,
            "frame_stack": 4,
            "maxpool_last_two": True,
            "obs_layout": "chw",
            "obs_copy": args.obs_copy,
            "info_filter": args.info_filter,
            "autoreset_mode": "Disabled",
            "action_policy": "fixed_noop",
            "warmup_steps": args.warmup_steps,
            "seconds": args.seconds,
        },
        "result": {
            "name": "retro_vec_stella",
            "runs": runs,
            "summary": _summary(values),
        },
        "runtime": {
            "platform": platform.platform(),
            "stable_retro_file": retro.__file__,
            "stable_retro_version": getattr(retro, "__version__", None),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="Breakout-Atari2600-v0")
    parser.add_argument("--state", default="Start")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--obs-copy", choices=("safe_view", "unsafe_view"), default="safe_view")
    parser.add_argument("--info-filter", choices=("all", "terminal", "none"), default="none")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.num_threads <= 0:
        raise SystemExit("num_envs and num_threads must be positive")
    if args.seconds <= 0 or args.repeats <= 0 or args.warmup_steps < 0:
        raise SystemExit("seconds/repeats must be positive and warmup_steps non-negative")

    result = _run(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"artifact={args.output_json}")
    summary = result["result"]["summary"]
    print(
        f"retro_vec_stella: mean={summary['mean']:.1f} "
        f"stdev={summary['stdev']:.1f} best={summary['max']:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

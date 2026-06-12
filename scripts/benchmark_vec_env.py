#!/usr/bin/env python3
"""Benchmark stable-retro vector rollout throughput variants."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Result:
    name: str
    steps: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


def _make_env_fn(
    game, state, inttype, env_kwargs, rom_path=None, info=None, scenario=None
):
    def make_env():
        import stable_retro as retro

        if rom_path is not None:
            retro.data.get_romfile_path = lambda *_args, **_kwargs: rom_path
            retro.data.get_file_path = lambda _game, file, *_args, **_kwargs: {
                "data.json": info,
                "scenario.json": scenario,
            }.get(file, file)
        return retro.make(game, state=state, inttype=inttype, **env_kwargs)

    return make_env


def _sample_actions(env):
    return np.asarray([env.action_space.sample() for _ in range(env.num_envs)])


def _run_vec(name, env, seconds, warmup_steps) -> Result:
    env.reset()
    for _ in range(warmup_steps):
        env.step(_sample_actions(env))
    steps = 0
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        env.step(_sample_actions(env))
        steps += env.num_envs
    elapsed = time.perf_counter() - start
    env.close()
    return Result(name=name, steps=steps, seconds=elapsed)


def _print_results(results):
    baseline = results[0].steps_per_second if results else 0.0
    print(
        f"{'variant':34} {'steps/s':>12} {'speedup':>10} {'steps':>10} {'seconds':>9}"
    )
    print("-" * 81)
    for result in results:
        speedup = result.steps_per_second / baseline if baseline else 0.0
        print(
            f"{result.name:34} "
            f"{result.steps_per_second:12.1f} "
            f"{speedup:10.2f} "
            f"{result.steps:10d} "
            f"{result.seconds:9.2f}",
        )


def _build_subproc(env_fns, start_method):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    return SubprocVecEnv(env_fns, start_method=start_method)


def _build_shared(env_fns, start_method):
    from stable_retro.vec_env import StableRetroSubprocVecEnv

    return StableRetroSubprocVecEnv(env_fns, start_method=start_method)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default=None)
    parser.add_argument("--rom-path", default=None)
    parser.add_argument("--info", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--resize", default="84x84")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--obs-crop", default=None)
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant names to run. Defaults to all variants.",
    )
    args = parser.parse_args(argv)

    import stable_retro as retro

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stable-retro")
    state = retro.State.DEFAULT if args.state is None else args.state
    game = args.game
    if args.rom_path is not None:
        rom_path = str(Path(args.rom_path).resolve())
        game = Path(rom_path).stem
        state = retro.State.NONE
        if args.info is None or args.scenario is None:
            raise SystemExit("--rom-path requires --info and --scenario")
        info = str(Path(args.info).resolve())
        scenario = str(Path(args.scenario).resolve())
    else:
        rom_path = None
        info = None
        scenario = None
    resize_h, resize_w = (int(v) for v in args.resize.lower().split("x", 1))
    obs_crop = None
    if args.obs_crop:
        obs_crop = tuple(int(v) for v in args.obs_crop.split(","))
        if len(obs_crop) != 4:
            raise SystemExit("--obs-crop must be top,bottom,left,right")

    base_kwargs = {
        "render_mode": "rgb_array",
    }
    preproc_kwargs = {
        **base_kwargs,
        "obs_resize": (resize_h, resize_w),
        "obs_grayscale": args.grayscale,
        "obs_crop": obs_crop,
        "frame_skip": args.frame_skip,
        "frame_stack": args.frame_stack,
        "maxpool_last_two": True,
    }

    variants = [
        ("subproc_baseline", _build_subproc, base_kwargs, False, False),
        ("subproc_worker_preproc", _build_subproc, preproc_kwargs, False, False),
        ("subproc_native_preproc", _build_subproc, preproc_kwargs, True, False),
        ("subproc_native_fused", _build_subproc, preproc_kwargs, True, True),
        ("shared_worker_preproc", _build_shared, preproc_kwargs, False, False),
        ("shared_native_preproc", _build_shared, preproc_kwargs, True, False),
        ("shared_native_fused", _build_shared, preproc_kwargs, True, True),
    ]
    if args.variants:
        requested_variants = {name.strip() for name in args.variants.split(",")}
        known_variants = {name for name, *_ in variants}
        unknown_variants = requested_variants - known_variants
        if unknown_variants:
            raise SystemExit(
                "Unknown variants: "
                + ", ".join(sorted(unknown_variants))
                + ". Known variants: "
                + ", ".join(name for name, *_ in variants),
            )
        variants = [variant for variant in variants if variant[0] in requested_variants]

    results = []
    old_disable_audio = os.environ.get("STABLE_RETRO_DISABLE_AUDIO")
    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    old_disable_native = os.environ.get("STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS")
    old_disable_fused = os.environ.get("STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP")
    try:
        for name, builder, kwargs, native, fused in variants:
            if native:
                os.environ.pop("STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS", None)
            else:
                os.environ["STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS"] = "1"
            if fused:
                os.environ.pop("STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP", None)
            else:
                os.environ["STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP"] = "1"
            env_fns = [
                _make_env_fn(
                    game,
                    state,
                    retro.data.Integrations.DEFAULT,
                    kwargs,
                    rom_path=rom_path,
                    info=info,
                    scenario=scenario,
                )
                for _ in range(args.num_envs)
            ]
            env = builder(env_fns, args.start_method)
            result = _run_vec(name, env, args.seconds, args.warmup_steps)
            results.append(result)
            print(
                f"finished {result.name}: "
                f"{result.steps_per_second:.1f} steps/s "
                f"({result.steps} steps in {result.seconds:.2f}s)",
                flush=True,
            )
    finally:
        if old_disable_audio is None:
            os.environ.pop("STABLE_RETRO_DISABLE_AUDIO", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_AUDIO"] = old_disable_audio
        if old_disable_native is None:
            os.environ.pop("STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS"] = old_disable_native
        if old_disable_fused is None:
            os.environ.pop("STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP"] = old_disable_fused

    _print_results(results)
    if len(results) > 1:
        best = max(results, key=lambda result: result.steps_per_second)
        speedup = best.steps_per_second / results[0].steps_per_second
        print(f"\nbest={best.name} speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

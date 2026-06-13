"""Benchmark the supported native stable-retro vector rollout path."""

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


def _build_native_vec(
    game,
    state,
    inttype,
    num_envs,
    env_kwargs,
    rom_path=None,
    info=None,
    scenario=None,
    num_threads=None,
    copy_observations=True,
):
    from stable_retro.vec_env import StableRetroNativeVecEnv

    return StableRetroNativeVecEnv(
        game,
        num_envs,
        state=state,
        inttype=inttype,
        rom_path=rom_path,
        info=info,
        scenario=scenario,
        num_threads=num_threads,
        copy_observations=copy_observations,
        **env_kwargs,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default=None)
    parser.add_argument("--rom-path", default=None)
    parser.add_argument("--info", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--resize", default="84x84")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--obs-crop", default=None)
    parser.add_argument("--copy-observations", action="store_true")
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

    env_kwargs = {
        "render_mode": "rgb_array",
        "obs_resize": (resize_h, resize_w),
        "obs_grayscale": args.grayscale,
        "obs_crop": obs_crop,
        "frame_skip": args.frame_skip,
        "frame_stack": args.frame_stack,
        "maxpool_last_two": True,
    }

    old_disable_audio = os.environ.get("STABLE_RETRO_DISABLE_AUDIO")
    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    try:
        env = _build_native_vec(
            game,
            state,
            retro.data.Integrations.DEFAULT,
            args.num_envs,
            env_kwargs,
            rom_path=rom_path,
            info=info,
            scenario=scenario,
            num_threads=args.num_threads,
            copy_observations=args.copy_observations,
        )
        result = _run_vec(
            "native_vec_fused",
            env,
            args.seconds,
            args.warmup_steps,
        )
    finally:
        if old_disable_audio is None:
            os.environ.pop("STABLE_RETRO_DISABLE_AUDIO", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_AUDIO"] = old_disable_audio

    print(
        f"{result.name}: {result.steps_per_second:.1f} steps/s "
        f"({result.steps} steps in {result.seconds:.2f}s)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

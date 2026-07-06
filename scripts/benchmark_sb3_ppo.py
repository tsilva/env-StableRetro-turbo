"""Benchmark end-to-end SB3 PPO throughput on stable-retro Mario.

This script is intentionally a throughput benchmark, not a recipe for solving
the game. It keeps the environment preprocessing fixed and measures PPO rollout
collection plus gradient updates after an excluded warmup section.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

from benchmark_vec_env import (
    _build_regular_vec,
    _default_profiles_json_path,
    _load_profiles,
    _parse_state,
    _resolve_backend,
)


DEFAULT_TRAIN_NUM_ENVS = 16
DEFAULT_TRAIN_NUM_THREADS = 4


@dataclass
class TimingSummary:
    warmup_updates: int
    measured_updates: int
    measured_steps: int
    rollout_seconds: float
    update_seconds: float
    total_seconds: float

    @property
    def train_steps_per_second(self) -> float:
        return self.measured_steps / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def rollout_steps_per_second(self) -> float:
        return (
            self.measured_steps / self.rollout_seconds
            if self.rollout_seconds > 0
            else 0.0
        )


def _parse_obs_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    crop = tuple(int(v) for v in value.split(","))
    if len(crop) != 4:
        raise SystemExit("--obs-crop must be top,bottom,left,right")
    return crop


def _parse_resize(value: str) -> tuple[int, int]:
    try:
        height, width = (int(v) for v in value.lower().split("x", 1))
    except ValueError as e:
        raise SystemExit("--resize must be formatted like 84x84") from e
    return height, width


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _source_version(retro) -> str:
    return str(getattr(retro, "__version__", "unknown")).strip()


def _configure_package_source(package_source: str) -> None:
    import sys

    repo_root = str(REPO_ROOT)
    if package_source == "checkout":
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return
    if package_source == "installed":
        sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != REPO_ROOT]
        return
    raise ValueError(f"Unsupported package source: {package_source}")


def _make_timing_callback(base_callback_cls):
    class TrainTimingCallback(base_callback_cls):
        def __init__(self, *, n_envs, n_steps, warmup_updates, measured_updates):
            super().__init__(verbose=0)
            self.n_envs = int(n_envs)
            self.n_steps = int(n_steps)
            self.warmup_updates = int(warmup_updates)
            self.target_measured_updates = int(measured_updates)
            self.completed_rollouts = 0
            self.measured_updates = 0
            self.measured_steps = 0
            self.rollout_seconds = 0.0
            self.update_seconds = 0.0
            self.measure_start = None
            self.measure_end = None
            self._rollout_start = None
            self._last_rollout_end = None
            self._measuring = False

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            now = time.perf_counter()
            if self._measuring and self._last_rollout_end is not None:
                self.update_seconds += now - self._last_rollout_end
                self._last_rollout_end = None

            if (
                not self._measuring
                and self.completed_rollouts >= self.warmup_updates
            ):
                self._measuring = True
                self.measure_start = now

            if self._measuring:
                self._rollout_start = now

        def _on_rollout_end(self) -> None:
            now = time.perf_counter()
            if self._measuring and self._rollout_start is not None:
                self.rollout_seconds += now - self._rollout_start
                self._last_rollout_end = now
                self.measured_updates += 1
                self.measured_steps += self.n_envs * self.n_steps
            self.completed_rollouts += 1

        def _on_training_end(self) -> None:
            now = time.perf_counter()
            if self._measuring and self._last_rollout_end is not None:
                self.update_seconds += now - self._last_rollout_end
                self._last_rollout_end = None
            self.measure_end = now

        def summary(self) -> TimingSummary:
            if self.measure_start is None or self.measure_end is None:
                total_seconds = 0.0
            else:
                total_seconds = self.measure_end - self.measure_start
            return TimingSummary(
                warmup_updates=self.warmup_updates,
                measured_updates=self.measured_updates,
                measured_steps=self.measured_steps,
                rollout_seconds=self.rollout_seconds,
                update_seconds=self.update_seconds,
                total_seconds=total_seconds,
            )

    return TrainTimingCallback


def _build_env(args, profile, retro):
    from stable_baselines3.common.vec_env import VecTransposeImage

    resize = profile.resize if args.resize is None else args.resize
    resize_h, resize_w = _parse_resize(resize)
    obs_crop_value = profile.obs_crop if args.obs_crop is None else args.obs_crop
    grayscale = profile.grayscale if args.grayscale is None else args.grayscale
    frame_skip = profile.frame_skip if args.frame_skip is None else args.frame_skip
    frame_stack = profile.frame_stack if args.frame_stack is None else args.frame_stack
    resize_algorithm = (
        profile.resize_algorithm
        if args.resize_algorithm is None
        else args.resize_algorithm
    )
    maxpool_last_two = profile.maxpool_last_two and not args.no_maxpool_last_two
    game = profile.game if args.game is None else args.game
    state_value = profile.state if args.state is None else args.state
    state = _parse_state(state_value, retro, allow_state_none=args.allow_state_none)

    env_kwargs = {
        "render_mode": "rgb_array",
        "obs_resize": (resize_h, resize_w),
        "obs_grayscale": grayscale,
        "obs_crop": _parse_obs_crop(obs_crop_value),
        "obs_resize_algorithm": resize_algorithm,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "maxpool_last_two": maxpool_last_two,
        "info_filter": args.info_filter,
        "obs_layout": args.obs_layout,
    }
    native_only_env_kwargs = {
        "noop_reset_max": args.noop_reset_max,
        "sticky_action_prob": args.sticky_action_prob,
        "reward_clip": args.reward_clip,
    }
    backend = _resolve_backend(args.backend)

    if backend == "native":
        from stable_retro.vec_env import RetroVecEnv

        env = RetroVecEnv(
            game,
            state=state,
            num_envs=args.num_envs,
            num_threads=args.num_threads,
            obs_copy=args.obs_copy,
            **env_kwargs,
            **native_only_env_kwargs,
        )
    else:
        if args.obs_copy != "safe_view":
            raise SystemExit("--obs-copy requires --backend=native")
        if any(native_only_env_kwargs.values()):
            raise SystemExit(
                "--noop-reset-max, --sticky-action-prob, and --reward-clip "
                "require --backend=native",
            )
        env = _build_regular_vec(
            backend,
            game,
            state,
            args.num_envs,
            env_kwargs,
            start_method=args.subproc_start_method,
        )
    if args.seed is not None:
        env.seed(args.seed)

    if args.vec_transpose_image:
        if args.obs_layout != "hwc":
            raise SystemExit("--vec-transpose-image requires --obs-layout=hwc")
        env = VecTransposeImage(env)

    resolved = {
        "profile": args.profile,
        "backend": backend,
        "game": game,
        "state": "State.NONE" if state is retro.State.NONE else str(state),
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "resize": resize,
        "grayscale": grayscale,
        "obs_crop": env_kwargs["obs_crop"],
        "resize_algorithm": resize_algorithm,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "maxpool_last_two": maxpool_last_two,
        "info_filter": args.info_filter,
        "obs_layout": args.obs_layout,
        "vec_transpose_image": args.vec_transpose_image,
        "obs_copy": args.obs_copy,
        "noop_reset_max": args.noop_reset_max,
        "sticky_action_prob": args.sticky_action_prob,
        "reward_clip": args.reward_clip,
        "subproc_start_method": args.subproc_start_method if backend != "native" else None,
    }
    return env, resolved


def _resolved_config(args, profile, retro) -> dict[str, Any]:
    resize = profile.resize if args.resize is None else args.resize
    obs_crop_value = profile.obs_crop if args.obs_crop is None else args.obs_crop
    grayscale = profile.grayscale if args.grayscale is None else args.grayscale
    frame_skip = profile.frame_skip if args.frame_skip is None else args.frame_skip
    frame_stack = profile.frame_stack if args.frame_stack is None else args.frame_stack
    resize_algorithm = (
        profile.resize_algorithm
        if args.resize_algorithm is None
        else args.resize_algorithm
    )
    game = profile.game if args.game is None else args.game
    state_value = profile.state if args.state is None else args.state
    state = _parse_state(state_value, retro, allow_state_none=args.allow_state_none)
    backend = _resolve_backend(args.backend)
    return {
        "profile": args.profile,
        "backend": backend,
        "game": game,
        "state": "State.NONE" if state is retro.State.NONE else str(state),
        "num_envs": args.num_envs,
        "num_threads": args.num_threads,
        "resize": resize,
        "grayscale": grayscale,
        "obs_crop": _parse_obs_crop(obs_crop_value),
        "resize_algorithm": resize_algorithm,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "maxpool_last_two": profile.maxpool_last_two and not args.no_maxpool_last_two,
        "info_filter": args.info_filter,
        "obs_layout": args.obs_layout,
        "vec_transpose_image": args.vec_transpose_image,
        "obs_copy": args.obs_copy,
        "noop_reset_max": args.noop_reset_max,
        "sticky_action_prob": args.sticky_action_prob,
        "reward_clip": args.reward_clip,
        "subproc_start_method": args.subproc_start_method if backend != "native" else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Short SB3 PPO training-throughput benchmark for stable-retro.",
    )
    parser.add_argument("--profiles-json", default=str(_default_profiles_json_path()))
    parser.add_argument("--profile", default="supermario-level1-1")
    parser.add_argument(
        "--package-source",
        choices=("checkout", "installed"),
        default="checkout",
        help=(
            "Import stable_retro from this checkout or from the active Python "
            "environment. Use 'installed' for wheel/post0 baselines."
        ),
    )
    parser.add_argument("--game", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_TRAIN_NUM_ENVS)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_TRAIN_NUM_THREADS)
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "subproc", "dummy"),
        default="auto",
    )
    parser.add_argument(
        "--subproc-start-method",
        choices=("fork", "forkserver", "spawn"),
        default="fork",
    )
    parser.add_argument("--resize", default=None)
    parser.add_argument("--grayscale", action="store_true", default=None)
    parser.add_argument("--rgb", action="store_false", dest="grayscale")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--obs-crop", default=None)
    parser.add_argument("--resize-algorithm", default=None)
    parser.add_argument(
        "--info-filter",
        choices=("terminal", "all", "none"),
        default="terminal",
    )
    parser.add_argument("--obs-layout", choices=("hwc", "chw"), default="hwc")
    parser.add_argument(
        "--vec-transpose-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Transpose HWC observations to CHW for SB3 CnnPolicy.",
    )
    parser.add_argument(
        "--obs-copy",
        choices=("copy", "safe_view", "unsafe_view"),
        default="safe_view",
    )
    parser.add_argument("--no-maxpool-last-two", action="store_true")
    parser.add_argument("--noop-reset-max", type=int, default=0)
    parser.add_argument("--sticky-action-prob", type=float, default=0.0)
    parser.add_argument("--reward-clip", action="store_true")
    parser.add_argument("--allow-state-none", action="store_true")
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measured-updates", type=int, default=5)
    parser.add_argument("--policy", default="CnnPolicy")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args(argv)

    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive")
    if args.num_threads is not None and args.num_threads <= 0:
        raise SystemExit("--num-threads must be positive")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.n_epochs <= 0:
        raise SystemExit("--n-epochs must be positive")
    if args.warmup_updates < 0:
        raise SystemExit("--warmup-updates must be non-negative")
    if args.measured_updates <= 0:
        raise SystemExit("--measured-updates must be positive")
    rollout_size = args.num_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        raise SystemExit(
            "--batch-size must divide --num-envs * --n-steps for stable timing",
        )
    if args.vec_transpose_image is None:
        args.vec_transpose_image = args.obs_layout == "hwc"

    profiles = _load_profiles(Path(args.profiles_json))
    try:
        profile = profiles[args.profile]
    except KeyError as e:
        available = ", ".join(sorted(profiles))
        raise SystemExit(
            f"Unknown benchmark profile {args.profile!r}. Available profiles: {available}",
        ) from e

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path("/tmp/matplotlib-stable-retro")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    _configure_package_source(args.package_source)
    import stable_retro as retro

    config = _resolved_config(args, profile, retro)
    ppo_config = {
        "policy": args.policy,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
    }
    run_config = {
        "env": config,
        "ppo": ppo_config,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "total_timesteps": rollout_size
        * (args.warmup_updates + args.measured_updates),
        "stable_retro_file": str(Path(retro.__file__).resolve()),
        "package_source": args.package_source,
        "stable_retro_source_version": _source_version(retro),
        "stable_retro_turbo_distribution_version": _package_version(
            "stable-retro-turbo",
        ),
    }
    print(json.dumps({"config": run_config}, sort_keys=True))
    if args.dry_run:
        return 0

    old_disable_audio = os.environ.get("STABLE_RETRO_DISABLE_AUDIO")
    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import set_random_seed

        if args.torch_num_threads is not None:
            torch.set_num_threads(args.torch_num_threads)
        if args.seed is not None:
            set_random_seed(args.seed)

        env, resolved_env = _build_env(args, profile, retro)
        callback_cls = _make_timing_callback(BaseCallback)
        callback = callback_cls(
            n_envs=args.num_envs,
            n_steps=args.n_steps,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
        )
        model = PPO(
            policy=args.policy,
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            seed=args.seed,
            verbose=0,
        )
        model.learn(
            total_timesteps=run_config["total_timesteps"],
            callback=callback,
            log_interval=None,
            progress_bar=False,
        )
        summary = callback.summary()
        result = {
            "env": resolved_env,
            "ppo": ppo_config,
            "timing": {
                "warmup_updates": summary.warmup_updates,
                "measured_updates": summary.measured_updates,
                "measured_steps": summary.measured_steps,
                "rollout_seconds": summary.rollout_seconds,
                "update_seconds": summary.update_seconds,
                "total_seconds": summary.total_seconds,
                "rollout_steps_per_second": summary.rollout_steps_per_second,
                "train_steps_per_second": summary.train_steps_per_second,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "stable_retro_file": str(Path(retro.__file__).resolve()),
                "package_source": args.package_source,
                "stable_retro_source_version": _source_version(retro),
                "stable_retro_turbo_distribution_version": _package_version(
                    "stable-retro-turbo",
                ),
                "stable_baselines3_version": _package_version("stable-baselines3"),
                "torch_version": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "device": str(model.device),
            },
        }
        print(
            "ppo_train: "
            f"{summary.train_steps_per_second:.1f} steps/s "
            f"({summary.measured_steps} measured steps in "
            f"{summary.total_seconds:.2f}s; rollout="
            f"{summary.rollout_seconds:.2f}s update={summary.update_seconds:.2f}s)",
        )
        print(json.dumps({"result": result}, sort_keys=True))
        if args.json_output is not None:
            Path(args.json_output).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            env.close()
        except UnboundLocalError:
            pass
        if old_disable_audio is None:
            os.environ.pop("STABLE_RETRO_DISABLE_AUDIO", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_AUDIO"] = old_disable_audio

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

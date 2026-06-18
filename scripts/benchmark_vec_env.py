"""Benchmark the supported native stable-retro vector rollout path."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    name: str
    steps: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    game: str
    state: str
    resize: str = "84x84"
    grayscale: bool = True
    frame_skip: int = 4
    frame_stack: int = 4
    obs_crop: str | None = None
    resize_algorithm: str = "area"
    maxpool_last_two: bool = True
    num_envs: int = 32
    num_threads: int | None = 16
    description: str = ""


def _default_profiles_json_path() -> Path:
    return Path(__file__).resolve().with_name("benchmark_vec_env.json")


def _load_profiles(path: Path) -> dict[str, BenchmarkProfile]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(
            f"Benchmark profile file not found: {path} (create it or pass --profiles-json)",
        ) from e

    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit(f"Benchmark profile file has no profiles: {path}")

    out: dict[str, BenchmarkProfile] = {}
    for i, item in enumerate(profiles):
        if not isinstance(item, dict):
            raise SystemExit(f"Invalid profile at index {i} in {path}")
        try:
            profile = BenchmarkProfile(
                name=str(item["name"]),
                game=str(item["game"]),
                state=str(item["state"]),
                resize=str(item.get("resize", "84x84")),
                grayscale=bool(item.get("grayscale", True)),
                frame_skip=int(item.get("frame_skip", 4)),
                frame_stack=int(item.get("frame_stack", 4)),
                obs_crop=(
                    None
                    if item.get("obs_crop") is None
                    else str(item.get("obs_crop"))
                ),
                resize_algorithm=str(item.get("resize_algorithm", "area")),
                maxpool_last_two=bool(item.get("maxpool_last_two", True)),
                num_envs=int(item.get("num_envs", 32)),
                num_threads=(
                    None
                    if item.get("num_threads") is None
                    else int(item.get("num_threads"))
                ),
                description=str(item.get("description", "")),
            )
        except KeyError as e:
            raise SystemExit(
                f"Missing key {e} in benchmark profile at index {i} in {path}",
            ) from e
        if profile.name in out:
            raise SystemExit(f"Duplicate benchmark profile name: {profile.name}")
        out[profile.name] = profile
    return out


def _parse_state(value, retro, *, allow_state_none: bool):
    normalized = str(value).strip()
    if normalized.lower() in {"none", "state.none"}:
        if not allow_state_none:
            raise SystemExit(
                "State.NONE benchmarks are disabled by default. Use an actual game state "
                "or pass --allow-state-none for low-level direct-ROM diagnostics.",
            )
        return retro.State.NONE
    if normalized.lower() in {"default", "state.default"}:
        return retro.State.DEFAULT
    return normalized


def _parse_info_keys(value, *, game, info, retro):
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() != "all":
        return [key.strip() for key in normalized.split(",") if key.strip()]

    if info is None:
        info_path = Path(
            retro.data.get_file_path(
                game,
                "data.json",
                retro.data.Integrations.DEFAULT,
            ),
        )
    else:
        info_path = Path(info)
    raw = json.loads(info_path.read_text(encoding="utf-8"))
    info_data = raw.get("info", {})
    if not isinstance(info_data, dict):
        raise SystemExit(f"Expected object-valued info in {info_path}")
    return sorted(str(key) for key in info_data)


def _sample_actions(env, fixed_actions=None):
    if fixed_actions is not None:
        return fixed_actions
    import numpy as np

    return np.asarray([env.action_space.sample() for _ in range(env.num_envs)])


def _run_vec(name, env, seconds, warmup_steps, fixed_actions=None) -> Result:
    env.reset()
    for _ in range(warmup_steps):
        env.step(_sample_actions(env, fixed_actions))
    steps = 0
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        env.step(_sample_actions(env, fixed_actions))
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
    parser.add_argument("--profiles-json", default=str(_default_profiles_json_path()))
    parser.add_argument("--profile", default="supermario-level1-1")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--game", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--rom-path", default=None)
    parser.add_argument("--info", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--resize", default=None)
    parser.add_argument("--grayscale", action="store_true", default=None)
    parser.add_argument("--rgb", action="store_false", dest="grayscale")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--obs-crop", default=None)
    parser.add_argument("--resize-algorithm", default=None)
    parser.add_argument(
        "--info-mode",
        choices=("terminal", "all", "none"),
        default="terminal",
    )
    parser.add_argument(
        "--info-keys",
        default=None,
        help="Comma-separated info keys to emit, or 'all' to pass all keys from data.json.",
    )
    parser.add_argument("--obs-layout", choices=("hwc", "chw"), default="hwc")
    parser.add_argument(
        "--vec-transpose-image",
        action="store_true",
        help="Wrap the native HWC env in SB3 VecTransposeImage to model PyTorch pixel training.",
    )
    parser.add_argument("--no-maxpool-last-two", action="store_true")
    parser.add_argument("--fixed-actions", action="store_true")
    parser.add_argument("--copy-observations", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved benchmark profile/config without creating envs.",
    )
    parser.add_argument(
        "--allow-state-none",
        action="store_true",
        help="Allow power-on/direct-ROM State.NONE benchmarks. Off by default.",
    )
    args = parser.parse_args(argv)

    profiles = _load_profiles(Path(args.profiles_json))
    if args.list_profiles:
        for name, profile in sorted(profiles.items()):
            suffix = f" - {profile.description}" if profile.description else ""
            print(f"{name}: {profile.game} state={profile.state}{suffix}")
        return 0
    try:
        profile = profiles[args.profile]
    except KeyError as e:
        available = ", ".join(sorted(profiles))
        raise SystemExit(
            f"Unknown benchmark profile {args.profile!r}. Available profiles: {available}",
        ) from e

    import stable_retro as retro

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-stable-retro")
    state_value = profile.state if args.state is None else args.state
    state = _parse_state(state_value, retro, allow_state_none=args.allow_state_none)
    game = profile.game if args.game is None else args.game
    if args.rom_path is not None:
        rom_path = str(Path(args.rom_path).resolve())
        if state is retro.State.NONE and not args.allow_state_none:
            raise SystemExit(
                "State.NONE benchmarks are disabled by default. Use an actual "
                "game state or pass --allow-state-none for low-level direct-ROM "
                "diagnostics.",
            )
        if state is retro.State.NONE and args.game is None:
            game = Path(rom_path).stem
        if state is retro.State.NONE and (args.info is None or args.scenario is None):
            raise SystemExit("--rom-path requires --info and --scenario")
        info = None if args.info is None else str(Path(args.info).resolve())
        scenario = None if args.scenario is None else str(Path(args.scenario).resolve())
    else:
        rom_path = None
        info = None
        scenario = None

    resize = profile.resize if args.resize is None else args.resize
    resize_h, resize_w = (int(v) for v in resize.lower().split("x", 1))
    obs_crop_value = profile.obs_crop if args.obs_crop is None else args.obs_crop
    obs_crop = None
    if obs_crop_value:
        obs_crop = tuple(int(v) for v in obs_crop_value.split(","))
        if len(obs_crop) != 4:
            raise SystemExit("--obs-crop must be top,bottom,left,right")
    grayscale = profile.grayscale if args.grayscale is None else args.grayscale
    frame_skip = profile.frame_skip if args.frame_skip is None else args.frame_skip
    frame_stack = profile.frame_stack if args.frame_stack is None else args.frame_stack
    num_envs = profile.num_envs if args.num_envs is None else args.num_envs
    num_threads = profile.num_threads if args.num_threads is None else args.num_threads
    resize_algorithm = (
        profile.resize_algorithm
        if args.resize_algorithm is None
        else args.resize_algorithm
    )
    maxpool_last_two = profile.maxpool_last_two and not args.no_maxpool_last_two
    info_keys = _parse_info_keys(args.info_keys, game=game, info=info, retro=retro)

    env_kwargs = {
        "render_mode": "rgb_array",
        "obs_resize": (resize_h, resize_w),
        "obs_grayscale": grayscale,
        "obs_crop": obs_crop,
        "obs_resize_algorithm": resize_algorithm,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "maxpool_last_two": maxpool_last_two,
        "info_mode": args.info_mode,
        "obs_layout": args.obs_layout,
    }
    if info_keys is not None:
        env_kwargs["info_keys"] = info_keys
    if args.vec_transpose_image and args.obs_layout != "hwc":
        raise SystemExit("--vec-transpose-image requires --obs-layout=hwc")

    state_label = "State.NONE" if state is retro.State.NONE else str(state)
    action_label = "fixed" if args.fixed_actions else "sampled"
    print(
        f"profile={args.profile} game={game} state={state_label} "
        f"envs={num_envs} threads={num_threads or num_envs} "
        f"resize={resize} grayscale={grayscale} crop={obs_crop} "
        f"resize_algorithm={resize_algorithm} frame_skip={frame_skip} "
        f"frame_stack={frame_stack} info_mode={args.info_mode} "
        f"info_keys={'default' if info_keys is None else len(info_keys)} "
        f"obs_layout={args.obs_layout} vec_transpose_image={args.vec_transpose_image} "
        f"actions={action_label}",
    )
    if args.dry_run:
        return 0

    old_disable_audio = os.environ.get("STABLE_RETRO_DISABLE_AUDIO")
    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    try:
        env = _build_native_vec(
            game,
            state,
            retro.data.Integrations.DEFAULT,
            num_envs,
            env_kwargs,
            rom_path=rom_path,
            info=info,
            scenario=scenario,
            num_threads=num_threads,
            copy_observations=args.copy_observations,
        )
        if args.vec_transpose_image:
            from stable_baselines3.common.vec_env import VecTransposeImage

            env = VecTransposeImage(env)
        fixed_actions = None
        if args.fixed_actions:
            fixed_actions = _sample_actions(env)
        result = _run_vec(
            "native_vec_fused",
            env,
            args.seconds,
            args.warmup_steps,
            fixed_actions=fixed_actions,
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

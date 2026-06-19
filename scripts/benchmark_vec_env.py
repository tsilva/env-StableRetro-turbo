"""Benchmark stable-retro vector rollout paths."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


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


def _add_rewards(left, right):
    if left is None:
        return right
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
        right,
        (list, tuple, np.ndarray),
    ):
        return (
            np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
        ).tolist()
    return left + right


class BenchmarkRetroPreprocessWrapper(gym.Wrapper):
    """Benchmark-only wrapper matching the native profile for classic RetroEnv."""

    def __init__(
        self,
        env,
        *,
        obs_resize,
        obs_crop,
        obs_grayscale,
        obs_resize_algorithm,
        frame_skip,
        frame_stack,
        maxpool_last_two,
    ):
        super().__init__(env)
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self._obs_resize = obs_resize
        self._obs_crop = obs_crop
        self._obs_grayscale = bool(obs_grayscale)
        self._obs_resize_algorithm = str(obs_resize_algorithm).lower()
        if self._obs_resize_algorithm == "box":
            self._obs_resize_algorithm = "area"
        if self._obs_resize_algorithm == "linear":
            self._obs_resize_algorithm = "bilinear"
        if self._obs_resize_algorithm not in {"nearest", "bilinear", "area"}:
            raise ValueError(
                "obs_resize_algorithm must be one of: nearest, bilinear, area",
            )
        self._frame_skip = int(frame_skip)
        self._frame_stack = int(frame_stack)
        self._maxpool_last_two = bool(maxpool_last_two)
        if self._frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if self._frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self._frame_stack_buffer = []
        sample = np.zeros(env.observation_space.shape, dtype=np.uint8)
        processed = self._process_observation(sample)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self._stacked_obs_shape(processed.shape),
            dtype=np.uint8,
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _stacked_obs_shape(self, shape):
        if self._frame_stack == 1:
            return shape
        if len(shape) == 1:
            return (shape[0] * self._frame_stack,)
        return (*shape[:-1], shape[-1] * self._frame_stack)

    def _stack_frames(self):
        if self._frame_stack == 1:
            return self._frame_stack_buffer[-1]
        if self._frame_stack_buffer[-1].ndim == 1:
            return np.concatenate(self._frame_stack_buffer, axis=0)
        return np.concatenate(self._frame_stack_buffer, axis=-1)

    def _reset_frame_stack(self, obs):
        obs = np.asarray(obs, dtype=np.uint8)
        if self._frame_stack == 1:
            self._frame_stack_buffer = [obs]
            return obs
        self._frame_stack_buffer = [obs.copy() for _ in range(self._frame_stack)]
        return self._stack_frames()

    def _append_frame_stack(self, obs):
        obs = np.asarray(obs, dtype=np.uint8)
        if not self._frame_stack_buffer:
            return self._reset_frame_stack(obs)
        self._frame_stack_buffer.append(obs.copy())
        del self._frame_stack_buffer[: -self._frame_stack]
        return self._stack_frames()

    def _apply_obs_crop(self, image):
        if self._obs_crop is None:
            return image
        top, bottom, left, right = self._obs_crop
        height, width = image.shape[:2]
        y2 = height - bottom if bottom else height
        x2 = width - right if right else width
        if top >= y2 or left >= x2:
            raise ValueError("obs_crop removes the entire observation")
        return image[top:y2, left:x2]

    def _apply_obs_grayscale(self, image):
        if image.ndim == 2:
            image = image[:, :, None]
        if not self._obs_grayscale:
            return image
        gray = (
            image[:, :, 0].astype(np.uint16) * 77
            + image[:, :, 1].astype(np.uint16) * 150
            + image[:, :, 2].astype(np.uint16) * 29
            + 128
        ) >> 8
        return gray.astype(np.uint8)[:, :, None]

    def _resize_obs(self, image):
        if self._obs_resize is None:
            return image
        height, width = self._obs_resize
        src_height, src_width = image.shape[:2]
        if self._obs_resize_algorithm == "nearest":
            y_idx = np.linspace(0, src_height - 1, height).astype(np.intp)
            x_idx = np.linspace(0, src_width - 1, width).astype(np.intp)
            return image[y_idx][:, x_idx]
        if self._obs_resize_algorithm == "bilinear":
            y = np.linspace(0, src_height - 1, height, dtype=np.float32)
            x = np.linspace(0, src_width - 1, width, dtype=np.float32)
            y0 = np.floor(y).astype(np.intp)
            x0 = np.floor(x).astype(np.intp)
            y1 = np.minimum(y0 + 1, src_height - 1)
            x1 = np.minimum(x0 + 1, src_width - 1)
            wy = (y - y0).astype(np.float32)[:, None, None]
            wx = (x - x0).astype(np.float32)[None, :, None]
            top = (
                image[y0][:, x0].astype(np.float32) * (1.0 - wx)
                + image[y0][:, x1].astype(np.float32) * wx
            )
            bottom = (
                image[y1][:, x0].astype(np.float32) * (1.0 - wx)
                + image[y1][:, x1].astype(np.float32) * wx
            )
            return np.clip(top * (1.0 - wy) + bottom * wy, 0, 255).astype(np.uint8)
        if height > src_height or width > src_width:
            raise ValueError("area resize only supports downscaling")
        y_edges = np.linspace(0, src_height, height + 1).astype(np.intp)
        x_edges = np.linspace(0, src_width, width + 1).astype(np.intp)
        y_edges[1:] = np.maximum(y_edges[1:], y_edges[:-1] + 1)
        x_edges[1:] = np.maximum(x_edges[1:], x_edges[:-1] + 1)
        y_edges[-1] = src_height
        x_edges[-1] = src_width
        integral = image.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)
        integral = np.pad(integral, ((1, 0), (1, 0), (0, 0)), mode="constant")
        y0 = y_edges[:-1]
        y1 = y_edges[1:]
        x0 = x_edges[:-1]
        x1 = x_edges[1:]
        sums = (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )
        pixels = ((y1 - y0)[:, None] * (x1 - x0)[None, :])[:, :, None]
        return (sums // pixels).astype(np.uint8)

    def _process_observation(self, obs):
        obs = np.asarray(obs, dtype=np.uint8)
        obs = self._apply_obs_crop(obs)
        obs = self._apply_obs_grayscale(obs)
        return self._resize_obs(obs)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._reset_frame_stack(self._process_observation(obs)), info

    def step(self, action):
        total_reward = None
        terminated = False
        truncated = False
        info = {}
        obs = None
        recent_obs = []
        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward = _add_rewards(total_reward, reward)
            if self._maxpool_last_two:
                recent_obs.append(np.asarray(obs, dtype=np.uint8))
                del recent_obs[:-2]
            if terminated or truncated:
                break
        if obs is None:
            raise RuntimeError("RetroEnv step did not produce an observation")
        if self._maxpool_last_two and len(recent_obs) == 2:
            obs = np.maximum(recent_obs[0], recent_obs[1])
        processed = self._process_observation(obs)
        reward = total_reward if total_reward is not None else 0.0
        return self._append_frame_stack(processed), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def _sample_actions(env, fixed_actions=None):
    if fixed_actions is not None:
        return fixed_actions

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


def _make_regular_retro_env(game, state, info, scenario, env_kwargs):
    import stable_retro as retro

    preprocess_kwargs = {
        "obs_resize": env_kwargs["obs_resize"],
        "obs_crop": env_kwargs["obs_crop"],
        "obs_grayscale": env_kwargs["obs_grayscale"],
        "obs_resize_algorithm": env_kwargs["obs_resize_algorithm"],
        "frame_skip": env_kwargs["frame_skip"],
        "frame_stack": env_kwargs["frame_stack"],
        "maxpool_last_two": env_kwargs["maxpool_last_two"],
    }
    env = retro.make(
        game,
        state=state,
        inttype=retro.data.Integrations.DEFAULT,
        info=info,
        scenario=scenario,
        render_mode=env_kwargs.get("render_mode", "rgb_array"),
    )
    return BenchmarkRetroPreprocessWrapper(env, **preprocess_kwargs)


def _build_regular_vec(
    backend,
    game,
    state,
    num_envs,
    env_kwargs,
    *,
    rom_path=None,
    info=None,
    scenario=None,
    start_method="fork",
):
    if rom_path is not None:
        raise SystemExit("--rom-path is only supported with --backend=native")
    if env_kwargs.get("info_keys") is not None:
        raise SystemExit("--info-keys is only supported with --backend=native")
    if env_kwargs.get("obs_layout", "hwc") != "hwc":
        raise SystemExit("--obs-layout=chw requires --backend=native")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        (
            lambda game=game, state=state, info=info, scenario=scenario: (
                _make_regular_retro_env(game, state, info, scenario, env_kwargs)
            )
        )
        for _ in range(num_envs)
    ]
    if backend == "dummy":
        return DummyVecEnv(env_fns)
    if backend == "subproc":
        return SubprocVecEnv(env_fns, start_method=start_method)
    raise ValueError(f"Unsupported regular backend: {backend}")


def _native_vec_available():
    try:
        from stable_retro.vec_env import StableRetroNativeVecEnv  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend(requested):
    if requested != "auto":
        return requested
    if _native_vec_available():
        return "native"
    return "subproc"


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
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "subproc", "dummy"),
        default="auto",
        help=(
            "Vector backend. 'native' uses StableRetroNativeVecEnv; 'subproc' and "
            "'dummy' use classic RetroEnv plus benchmark preprocessing wrappers. "
            "'auto' chooses native when available, otherwise subproc for vanilla "
            "post0-style builds."
        ),
    )
    parser.add_argument(
        "--subproc-start-method",
        choices=("fork", "forkserver", "spawn"),
        default="fork",
        help="Multiprocessing start method for --backend=subproc.",
    )
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
    backend = _resolve_backend(args.backend)

    state_label = "State.NONE" if state is retro.State.NONE else str(state)
    action_label = "fixed" if args.fixed_actions else "sampled"
    parallel_label = (
        f"threads={num_threads or num_envs}"
        if backend == "native"
        else f"workers={num_envs}"
    )
    print(
        f"profile={args.profile} backend={backend} game={game} state={state_label} "
        f"envs={num_envs} {parallel_label} "
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
        if backend == "native":
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
            result_name = "native_vec_fused"
        else:
            env = _build_regular_vec(
                backend,
                game,
                state,
                num_envs,
                env_kwargs,
                rom_path=rom_path,
                info=info,
                scenario=scenario,
                start_method=args.subproc_start_method,
            )
            result_name = f"{backend}_vec_retro"
        if args.vec_transpose_image:
            from stable_baselines3.common.vec_env import VecTransposeImage

            env = VecTransposeImage(env)
        fixed_actions = None
        if args.fixed_actions:
            fixed_actions = _sample_actions(env)
        result = _run_vec(
            result_name,
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

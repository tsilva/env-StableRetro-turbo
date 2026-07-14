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

MARIO_SIMPLE_ACTIONS = {
    "noop": (0, 0, 0, 0, 0, 0, 0, 0, 0),
    "right": (0, 0, 0, 0, 0, 0, 0, 1, 0),
    "right_b": (1, 0, 0, 0, 0, 0, 0, 1, 0),
    "right_a": (0, 0, 0, 0, 0, 0, 0, 1, 1),
    "right_a_b": (1, 0, 0, 0, 0, 0, 0, 1, 1),
    "a": (0, 0, 0, 0, 0, 0, 0, 0, 1),
    "left": (0, 0, 0, 0, 0, 0, 1, 0, 0),
}


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
    obs_layout: str = "hwc"
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
                obs_layout=str(item.get("obs_layout", "hwc")).lower(),
                description=str(item.get("description", "")),
            )
        except KeyError as e:
            raise SystemExit(
                f"Missing key {e} in benchmark profile at index {i} in {path}",
            ) from e
        if profile.name in out:
            raise SystemExit(f"Duplicate benchmark profile name: {profile.name}")
        if profile.obs_layout not in {"hwc", "chw"}:
            raise SystemExit(
                f"Invalid obs_layout for profile {profile.name!r}: {profile.obs_layout!r}",
            )
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


def _resolve_game(profile_game: str, game_override: str | None, platform: str | None) -> str:
    game = profile_game if game_override is None else game_override.strip()
    if not game:
        raise SystemExit("--game must not be empty")
    if platform is None:
        return game

    platform = platform.strip()
    if not platform:
        raise SystemExit("--platform must not be empty")
    if game_override is None:
        raise SystemExit("--platform requires --game")

    platform_suffix = f"-{platform}-v0"
    if game.endswith("-v0"):
        if not game.endswith(platform_suffix):
            raise SystemExit(
                f"--game {game!r} does not match --platform {platform!r}; "
                f"pass the short game name or omit --platform",
            )
        return game
    return f"{game}{platform_suffix}"


def _parse_info_filter_keys(value, *, game, info, retro):
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
        obs_crop_mode,
        obs_crop_fill,
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
        self._obs_crop_mode = str(obs_crop_mode).lower()
        self._obs_crop_fill = int(obs_crop_fill)
        if self._obs_crop_mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        if not 0 <= self._obs_crop_fill <= 255:
            raise ValueError("obs_crop_fill must be between 0 and 255")
        self._obs_resize_algorithm = str(obs_resize_algorithm).lower()
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
        if self._obs_crop_mode == "mask":
            masked = image.copy()
            if top:
                masked[:top, :] = self._obs_crop_fill
            if bottom:
                masked[y2:, :] = self._obs_crop_fill
            if left:
                masked[:, :left] = self._obs_crop_fill
            if right:
                masked[:, x2:] = self._obs_crop_fill
            return masked
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

    action_space = getattr(env, "single_action_space", env.action_space)
    return np.asarray([action_space.sample() for _ in range(env.num_envs)])


def _parse_actions(value):
    if value is None:
        return None
    actions = [item.strip() for item in str(value).split(",")]
    if not actions or not all(actions):
        raise SystemExit("--actions must be a comma-separated list without empty entries")
    unknown = sorted(set(actions) - set(MARIO_SIMPLE_ACTIONS))
    if unknown:
        available = ", ".join(sorted(MARIO_SIMPLE_ACTIONS))
        raise SystemExit(
            f"Unknown benchmark action(s): {', '.join(unknown)}. "
            f"Available actions: {available}",
        )
    return tuple(actions)


def _action_templates(action_names, num_envs):
    return tuple(
        np.repeat(
            np.asarray(MARIO_SIMPLE_ACTIONS[name], dtype=np.uint8)[None, :],
            num_envs,
            axis=0,
        )
        for name in action_names
    )


def _sample_action_sequence(templates, count, seed):
    if count <= 0:
        return ()
    if len(templates) == 1:
        return tuple(templates[0] for _ in range(count))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(templates), size=count)
    return tuple(templates[int(index)] for index in indices)


def _expand_round_robin_states(states, num_envs):
    if states is None or len(states) >= num_envs:
        return states
    return [states[index % len(states)] for index in range(num_envs)]


def _step_and_reset(env, action, manual_reset):
    result = env.step(action)
    if manual_reset:
        done = np.asarray(result[2], dtype=bool) | np.asarray(result[3], dtype=bool)
        if done.any():
            env.reset(options={"reset_mask": done})
    return result


def _run_vec(
    name,
    env,
    seconds,
    warmup_steps,
    fixed_actions=None,
    manual_reset=False,
) -> Result:
    env.reset()
    for _ in range(warmup_steps):
        _step_and_reset(env, _sample_actions(env, fixed_actions), manual_reset)
    steps = 0
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        _step_and_reset(env, _sample_actions(env, fixed_actions), manual_reset)
        steps += env.num_envs
    elapsed = time.perf_counter() - start
    env.close()
    return Result(name=name, steps=steps, seconds=elapsed)


def _run_vec_steps(
    name,
    env,
    steps,
    repeats,
    warmup_steps,
    action_names,
    action_seed,
    manual_reset=False,
):
    if steps <= 0:
        raise SystemExit("--steps must be positive")
    if repeats <= 0:
        raise SystemExit("--repeats must be positive")
    templates = _action_templates(action_names, env.num_envs)
    warmup_actions = _sample_action_sequence(templates, warmup_steps, action_seed + 1)
    measured_actions = _sample_action_sequence(templates, steps, action_seed)
    env.reset()
    for action in warmup_actions:
        _step_and_reset(env, action, manual_reset)

    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        for action in measured_actions:
            _step_and_reset(env, action, manual_reset)
        elapsed = time.perf_counter() - start
        results.append(Result(name=name, steps=steps * env.num_envs, seconds=elapsed))
    env.close()
    return results


def _build_native_vec(
    game,
    state,
    states,
    state_probs,
    inttype,
    num_envs,
    env_kwargs,
    rom_path=None,
    info=None,
    scenario=None,
    num_threads=None,
    obs_copy="copy",
):
    from stable_retro.vec_env import RetroVecEnv

    state_arg = state
    if states is not None:
        state_arg = (
            dict(zip(states, state_probs, strict=True))
            if state_probs is not None
            else list(states)
        )

    return RetroVecEnv(
        game,
        state=state_arg,
        num_envs=num_envs,
        inttype=inttype,
        rom_path=rom_path,
        info=info,
        scenario=scenario,
        num_threads=num_threads,
        obs_copy=obs_copy,
        **env_kwargs,
    )


def _make_regular_retro_env(game, state, info, scenario, env_kwargs):
    import stable_retro as retro

    preprocess_kwargs = {
        "obs_resize": env_kwargs["obs_resize"],
        "obs_crop": env_kwargs["obs_crop"],
        "obs_grayscale": env_kwargs["obs_grayscale"],
        "obs_resize_algorithm": env_kwargs["obs_resize_algorithm"],
        "obs_crop_mode": env_kwargs["obs_crop_mode"],
        "obs_crop_fill": env_kwargs["obs_crop_fill"],
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
    info_filter = env_kwargs.get("info_filter")
    if isinstance(info_filter, dict) and info_filter.get("keys") is not None:
        raise SystemExit("--info-filter-keys is only supported with --backend=native")
    if env_kwargs.get("obs_layout", "hwc") != "hwc":
        raise SystemExit("--obs-layout=chw requires --backend=native")
    env_fns = [
        (
            lambda game=game, state=state, info=info, scenario=scenario: (
                _make_regular_retro_env(game, state, info, scenario, env_kwargs)
            )
        )
        for _ in range(num_envs)
    ]
    if backend == "dummy":
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv(env_fns)
    if backend == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(env_fns, start_method=start_method)
    if backend == "async":
        from gymnasium.vector import AsyncVectorEnv

        return AsyncVectorEnv(env_fns, context=start_method)
    raise ValueError(f"Unsupported regular backend: {backend}")


def _native_vec_available():
    try:
        from stable_retro.vec_env import RetroVecEnv  # noqa: F401
        from stable_retro import _retro
    except ImportError:
        return False
    return hasattr(_retro, "_RetroVecEnv")


def _sb3_vec_available():
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend(requested):
    if requested != "auto":
        return requested
    if _native_vec_available():
        return "native"
    if _sb3_vec_available():
        return "subproc"
    return "async"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-json", default=str(_default_profiles_json_path()))
    parser.add_argument("--profile", default="supermario-level1-1")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--game", default=None)
    parser.add_argument(
        "--platform",
        default=None,
        help=(
            "Stable Retro platform suffix to combine with a short --game name, "
            "for example --game MegaMan --platform Nes -> MegaMan-Nes-v0.",
        ),
    )
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated native start states. Without --state-probs, count must match --num-envs.",
    )
    parser.add_argument(
        "--state-probs",
        default=None,
        help="Comma-separated positive probabilities for --states; normalized before sampling.",
    )
    parser.add_argument("--rom-path", default=None)
    parser.add_argument("--info", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "subproc", "dummy", "async"),
        default="auto",
        help=(
            "Vector backend. 'native' uses RetroVecEnv; 'subproc' and "
            "'dummy' use classic RetroEnv plus benchmark preprocessing wrappers. "
            "'async' uses Gymnasium AsyncVectorEnv with the same wrappers. "
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Run a fixed number of vector steps per repeat instead of a seconds loop.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--resize", default=None)
    parser.add_argument("--grayscale", action="store_true", default=None)
    parser.add_argument("--rgb", action="store_false", dest="grayscale")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--obs-crop", default=None)
    parser.add_argument("--obs-crop-mode", choices=("remove", "mask"), default="remove")
    parser.add_argument("--obs-crop-fill", type=int, default=0)
    parser.add_argument("--resize-algorithm", default=None)
    parser.add_argument(
        "--info-filter",
        choices=("terminal", "all", "none"),
        default="terminal",
    )
    parser.add_argument(
        "--info-filter-keys",
        default=None,
        help="Comma-separated info keys to emit, or 'all' to pass all keys from data.json.",
    )
    parser.add_argument("--obs-layout", choices=("hwc", "chw"), default=None)
    parser.add_argument(
        "--vec-transpose-image",
        action="store_true",
        help="Wrap the native HWC env in SB3 VecTransposeImage to model PyTorch pixel training.",
    )
    parser.add_argument("--no-maxpool-last-two", action="store_true")
    parser.add_argument("--fixed-actions", action="store_true")
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated named Mario actions for fixed-step mode.",
    )
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument(
        "--obs-copy",
        choices=("copy", "safe_view", "unsafe_view"),
        default="safe_view",
    )
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
    if args.state is not None and args.states is not None:
        raise SystemExit("--state and --states are mutually exclusive")
    if args.state_probs is not None and args.states is None:
        raise SystemExit("--state-probs requires --states")
    states = None
    state_probs = None
    if args.states is not None:
        states = [item.strip() for item in args.states.split(",")]
        if not all(states):
            raise SystemExit("--states must not contain empty entries")
        if args.state_probs is not None:
            try:
                state_probs = [
                    float(item.strip()) for item in args.state_probs.split(",")
                ]
            except ValueError as e:
                raise SystemExit("--state-probs must be comma-separated numbers") from e
        else:
            states = _expand_round_robin_states(
                states,
                profile.num_envs if args.num_envs is None else args.num_envs,
            )
        state = None
    else:
        state_value = profile.state if args.state is None else args.state
        state = _parse_state(state_value, retro, allow_state_none=args.allow_state_none)
    game = _resolve_game(profile.game, args.game, args.platform)
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
    obs_layout = profile.obs_layout if args.obs_layout is None else args.obs_layout
    info_filter_keys = _parse_info_filter_keys(
        args.info_filter_keys,
        game=game,
        info=info,
        retro=retro,
    )

    env_kwargs = {
        "render_mode": "rgb_array",
        "obs_resize": (resize_h, resize_w),
        "obs_grayscale": grayscale,
        "obs_crop": obs_crop,
        "obs_crop_mode": args.obs_crop_mode,
        "obs_crop_fill": args.obs_crop_fill,
        "obs_resize_algorithm": resize_algorithm,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "maxpool_last_two": maxpool_last_two,
        "info_filter": args.info_filter,
        "obs_layout": obs_layout,
    }
    if info_filter_keys is not None:
        env_kwargs["info_filter"] = {
            "mode": args.info_filter,
            "keys": info_filter_keys,
        }
    if args.vec_transpose_image and obs_layout != "hwc":
        raise SystemExit("--vec-transpose-image requires --obs-layout=hwc")
    backend = _resolve_backend(args.backend)
    if states is not None and backend != "native":
        raise SystemExit("--states requires --backend=native")
    action_names = _parse_actions(args.actions)
    if args.steps is not None and action_names is None:
        raise SystemExit("--steps requires --actions for deterministic fixed-step mode")

    if states is not None:
        state_label = ",".join(states)
        if state_probs is None:
            state_label += " slot-assigned"
        else:
            state_label += f" probs={state_probs}"
    else:
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
        f"crop_mode={args.obs_crop_mode} resize_algorithm={resize_algorithm} "
        f"frame_skip={frame_skip} frame_stack={frame_stack} "
        f"maxpool_last_two={maxpool_last_two} "
        f"info_filter={args.info_filter} "
        f"info_filter_keys={'default' if info_filter_keys is None else len(info_filter_keys)} "
        f"obs_layout={obs_layout} vec_transpose_image={args.vec_transpose_image} "
        f"obs_copy={args.obs_copy} actions={action_names or action_label} "
        f"autoreset_mode={'Disabled' if backend == 'native' else 'backend-managed'} "
        f"steps={args.steps} repeats={args.repeats} seconds={args.seconds}",
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
                states,
                state_probs,
                retro.data.Integrations.DEFAULT,
                num_envs,
                env_kwargs,
                rom_path=rom_path,
                info=info,
                scenario=scenario,
                num_threads=num_threads,
                obs_copy=args.obs_copy,
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
        if args.vec_transpose_image and backend == "native":
            raise SystemExit(
                "--vec-transpose-image is an SB3 VecEnv wrapper; native "
                "RetroVecEnv is Gymnasium-first. Use obs_layout=chw or a "
                "downstream SB3 adapter.",
            )
        if args.vec_transpose_image:
            from stable_baselines3.common.vec_env import VecTransposeImage

            env = VecTransposeImage(env)
        if args.steps is None:
            fixed_actions = None
            if args.fixed_actions:
                fixed_actions = _sample_actions(env)
            result = _run_vec(
                result_name,
                env,
                args.seconds,
                args.warmup_steps,
                fixed_actions=fixed_actions,
                manual_reset=backend == "native",
            )
            results = [result]
        else:
            results = _run_vec_steps(
                result_name,
                env,
                args.steps,
                args.repeats,
                args.warmup_steps,
                action_names,
                args.action_seed,
                manual_reset=backend == "native",
            )
    finally:
        if old_disable_audio is None:
            os.environ.pop("STABLE_RETRO_DISABLE_AUDIO", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_AUDIO"] = old_disable_audio

    for index, result in enumerate(results, start=1):
        print(
            f"run={index} {result.name}: {result.steps_per_second:.1f} steps/s "
            f"({result.steps} steps in {result.seconds:.2f}s)",
        )
    if len(results) > 1:
        values = [result.steps_per_second for result in results]
        stdev = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        print(
            f"summary={results[0].name} steps_per_sec_mean={float(np.mean(values)):.1f} "
            f"steps_per_sec_stdev={stdev:.1f} "
            f"best_steps_per_sec={float(np.max(values)):.1f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

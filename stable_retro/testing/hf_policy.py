"""Helpers for validating Hugging Face SB3 policies against RetroVecEnv."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import stable_retro as retro
from stable_retro.vec_env import RetroVecEnv


MARIO_SIMPLE_ACTIONS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # noop
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
        [1, 0, 0, 0, 0, 0, 0, 1, 0],  # right_b
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right_a
        [1, 0, 0, 0, 0, 0, 0, 1, 1],  # right_a_b
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class PolicyEventResult:
    event_name: str
    episode: int
    step: int
    action: int
    payload: dict
    info: dict


def resolve_hf_policy_path(
    repo_id: str,
    filename: str,
    *,
    env_var: str | None = None,
    local_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a policy checkpoint from an env override or Hugging Face Hub."""

    if env_var:
        override = os.environ.get(env_var)
        if override:
            path = Path(override)
            if path.exists():
                return path
            raise FileNotFoundError(f"{env_var} points to missing file: {path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "huggingface_hub is required unless a local policy path is supplied",
        ) from e

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=None if local_dir is None else str(local_dir),
    )
    return Path(path)


def load_sb3_policy(path: str | os.PathLike[str], *, device: str = "cpu"):
    """Load an SB3 policy checkpoint for inference."""

    if importlib.util.find_spec("numpy._core.numeric") is None:
        raise RuntimeError(
            "this checkpoint requires NumPy 2-compatible pickle module paths",
        )
    try:
        from stable_baselines3 import PPO
    except ImportError as e:  # pragma: no cover - optional dependency.
        raise RuntimeError("stable_baselines3 is required to load SB3 policies") from e
    try:
        return PPO.load(str(path), device=device)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"failed to deserialize SB3 policy: {e}") from e


def make_mario_level1_policy_env(*, done_on, info_filter="terminal") -> RetroVecEnv:
    """Create the Super Mario Bros Level1-1 env shape used by the HF policy."""

    rom_path = retro.data.get_original_romfile_path("SuperMarioBros-Nes-v0")
    return RetroVecEnv(
        "SuperMarioBros-Nes-v0",
        state="Level1-1",
        num_envs=1,
        rom_path=rom_path,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        obs_crop=(32, 0, 0, 0),
        obs_resize=(84, 84),
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=True,
        obs_layout="chw",
        num_threads=1,
        info_filter=info_filter,
        done_on=done_on,
    )


def run_policy_until_event(
    model,
    env: RetroVecEnv,
    *,
    event_name: str,
    action_map: np.ndarray = MARIO_SIMPLE_ACTIONS,
    episodes: int = 10,
    max_steps: int = 2500,
    seed_start: int = 10007,
    deterministic: bool = False,
) -> PolicyEventResult | None:
    """Run a discrete SB3 policy until a named done_on event fires."""

    for episode in range(int(episodes)):
        env.seed(int(seed_start) + episode)
        obs = env.reset()
        for step in range(1, int(max_steps) + 1):
            action, _state = model.predict(obs, deterministic=deterministic)
            action_value = int(np.asarray(action).reshape(-1)[0])
            masks = action_map[np.asarray(action, dtype=np.int64).reshape(-1)]
            obs, _rewards, dones, infos = env.step(masks)
            if not bool(dones[0]):
                continue

            info = infos[0]
            payload = info.get("done_on_info", {})
            if event_name not in payload:
                break
            return PolicyEventResult(
                event_name=event_name,
                episode=episode,
                step=step,
                action=action_value,
                payload=payload[event_name],
                info=info,
            )
    return None

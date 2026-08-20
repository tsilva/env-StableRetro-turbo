from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HF_LEVEL1_POLICY = "tsilva/SuperMarioBros-NES_Level1"
HF_LEVEL1_POLICY_FILENAME = "ppo_supermariobros-nes-v0_4500000_steps.zip"
MAX_EPISODES = 10
MAX_STEPS_PER_EPISODE = 3_000
RUNNER = r"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

GAME = "SuperMarioBros-Nes-v0"
STATE = "Level1-1"
ACTION_BUTTONS = {
    "noop": (),
    "right": ("RIGHT",),
    "right_b": ("RIGHT", "B"),
    "right_a": ("RIGHT", "A"),
    "right_a_b": ("RIGHT", "A", "B"),
    "a": ("A",),
    "left": ("LEFT",),
}
ACTION_NAMES = tuple(ACTION_BUTTONS)


def emit(payload):
    print(json.dumps(payload, sort_keys=True))


def import_retro(package):
    if package == "env-stableretro-turbo":
        import env_stableretro_turbo as retro
    elif package == "stable-retro":
        import retro
    else:
        raise ValueError(package)
    return retro


def unavailable(reason):
    emit({"status": "unavailable", "reason": reason})


def probe(package):
    try:
        retro = import_retro(package)
    except Exception as exc:
        unavailable(f"{package} import failed: {type(exc).__name__}: {exc}")
        return
    try:
        states = retro.data.list_states(GAME)
        rom_path = retro.data.get_romfile_path(GAME)
    except Exception as exc:
        unavailable(f"{package} Super Mario data unavailable: {type(exc).__name__}: {exc}")
        return
    if STATE not in states:
        unavailable(f"{package} missing {GAME} {STATE} state")
        return
    if not Path(rom_path).exists():
        unavailable(f"{package} ROM path does not exist: {rom_path}")
        return
    emit(
        {
            "status": "ok",
            "package": package,
            "version": str(getattr(retro, "__version__", "")).strip(),
            "rom_path": str(rom_path),
        },
    )


def normalize(value):
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def array_hash(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def area_resize_gray(image, out_height=84, out_width=84):
    src = np.asarray(image, dtype=np.uint8)
    src_height, src_width = src.shape[:2]
    y_edges = np.linspace(0, src_height, out_height + 1).astype(np.intp)
    x_edges = np.linspace(0, src_width, out_width + 1).astype(np.intp)
    y_edges[1:] = np.maximum(y_edges[1:], y_edges[:-1] + 1)
    x_edges[1:] = np.maximum(x_edges[1:], x_edges[:-1] + 1)
    y_edges[-1] = src_height
    x_edges[-1] = src_width
    integral = src.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
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
    pixels = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (sums // pixels).astype(np.uint8)


def preprocess_screen(screen):
    screen = np.asarray(screen, dtype=np.uint8)
    cropped = screen[32:, :, :]
    gray = (
        cropped[:, :, 0].astype(np.uint16) * 77
        + cropped[:, :, 1].astype(np.uint16) * 150
        + cropped[:, :, 2].astype(np.uint16) * 29
        + 128
    ) >> 8
    return area_resize_gray(gray)


def stacked_obs(stack):
    return np.stack(stack, axis=0)


def action_masks(retro, rom_path):
    system = retro.get_romfile_system(str(rom_path))
    core = retro.get_system_info(system)
    buttons = tuple(None if name is None else str(name).upper() for name in core["buttons"])
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    masks = np.zeros((len(ACTION_NAMES), len(buttons)), dtype=np.uint8)
    for action_index, action_name in enumerate(ACTION_NAMES):
        for button in ACTION_BUTTONS[action_name]:
            masks[action_index, button_to_index[button]] = 1
    return masks


def level_was_cleared(info):
    if bool(info.get("level_complete")) or bool(info.get("completion_event")):
        return True

    return int(info.get("levelLo", 0)) > 0


def load_policy(args):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        unavailable(f"stable_baselines3 is not installed: {exc}")
        return None
    policy_path = os.environ.get("STABLE_RETRO_POLICY_PATH")
    try:
        if policy_path:
            resolved = Path(policy_path).expanduser()
        else:
            from huggingface_hub import hf_hub_download

            resolved = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=args.filename,
                    cache_dir=str(args.cache_dir),
                ),
            )
        zero_schedule = lambda _progress_remaining: 0.0
        return PPO.load(
            str(resolved),
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": zero_schedule,
                "clip_range": 0.15,
                "clip_range_vf": None,
            },
        )
    except Exception as exc:
        unavailable(f"policy load failed: {type(exc).__name__}: {exc}")
        return None


def make_env(package, retro, rom_path):
    if package == "env-stableretro-turbo":
        return retro.RetroVecEnv(
            GAME,
            state=STATE,
            num_envs=1,
            num_threads=1,
            rom_path=str(rom_path),
            render_mode="rgb_array",
            use_restricted_actions=retro.Actions.ALL,
            obs_crop=(32, 0, 0, 0),
            obs_resize=(84, 84),
            obs_grayscale=True,
            obs_resize_algorithm="area",
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
            noop_reset_max=0,
            sticky_action_prob=0.0,
            reward_clip=False,
            info_filter="all",
            obs_layout="chw",
            obs_copy="safe_view",
        )

    return retro.make(
        GAME,
        state=STATE,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
    )


def reset_env(package, env, seed):
    if package == "env-stableretro-turbo":
        if hasattr(env, "seed"):
            env.seed(seed)
        obs_batch = env.reset()
        return np.asarray(obs_batch[0], dtype=np.uint8), {}

    raw_obs, reset_info = env.reset(seed=seed)
    first = preprocess_screen(raw_obs)
    stack = [first.copy() for _ in range(4)]
    return stacked_obs(stack), reset_info


def step_env(package, env, mask, action_id, raw_stack):
    if package == "env-stableretro-turbo":
        obs_batch, rewards, dones, infos = env.step(mask[None, :])
        return (
            np.asarray(obs_batch[0], dtype=np.uint8),
            float(np.asarray(rewards).reshape(-1)[0]),
            bool(np.asarray(dones).reshape(-1)[0]),
            False,
            dict(infos[0]),
            raw_stack,
        )

    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    raw_frames = []
    for _ in range(4):
        raw_obs, frame_reward, frame_terminated, frame_truncated, frame_info = env.step(mask)
        raw_frames.append(np.asarray(raw_obs, dtype=np.uint8))
        reward += float(frame_reward)
        terminated = bool(frame_terminated)
        truncated = bool(frame_truncated)
        info = dict(frame_info)
        if terminated or truncated:
            break
    if len(raw_frames) >= 2:
        next_screen = np.maximum(raw_frames[-2], raw_frames[-1])
    else:
        next_screen = raw_frames[-1]
    raw_stack.append(preprocess_screen(next_screen))
    raw_stack = raw_stack[-4:]
    return (
        stacked_obs(raw_stack),
        reward,
        terminated,
        truncated,
        info,
        raw_stack,
    )


def run_trace(args):
    try:
        retro = import_retro(args.package)
        rom_path = retro.data.get_romfile_path(GAME)
        masks = action_masks(retro, rom_path)
        model = load_policy(args)
        if model is None:
            return
        env = make_env(args.package, retro, rom_path)
    except Exception as exc:
        unavailable(f"{args.package} setup failed: {type(exc).__name__}: {exc}")
        return

    trace = []
    episode_summaries = []
    completed = False
    try:
        for episode in range(1, args.episodes + 1):
            obs, reset_info = reset_env(args.package, env, args.seed + episode - 1)
            stack = [obs[index].copy() for index in range(obs.shape[0])]
            trace.append(
                {
                    "phase": "reset",
                    "episode": episode,
                    "obs": array_hash(obs),
                    "info": normalize(reset_info),
                },
            )
            final_info = {}
            final_reward = 0.0
            final_step = 0
            for step in range(1, args.steps + 1):
                action, _state = model.predict(obs[None, ...], deterministic=True)
                action_id = int(np.asarray(action).reshape(-1)[0])
                mask = masks[action_id]
                obs, reward, terminated, truncated, info, stack = step_env(
                    args.package,
                    env,
                    mask,
                    action_id,
                    stack,
                )
                trace.append(
                    {
                        "phase": "step",
                        "episode": episode,
                        "step": step,
                        "action": action_id,
                        "action_name": ACTION_NAMES[action_id],
                        "obs": array_hash(obs),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": normalize(info),
                    },
                )
                final_info = info
                final_reward = reward
                final_step = step
                completed = level_was_cleared(final_info)
                if completed or terminated or truncated:
                    break
            episode_summaries.append(
                {
                    "episode": episode,
                    "steps": final_step,
                    "reward": final_reward,
                    "completed": completed,
                    "final_info": normalize(final_info),
                },
            )
            if completed:
                break
    finally:
        env.close()

    emit(
        {
            "status": "ok",
            "package": args.package,
            "version": str(getattr(retro, "__version__", "")).strip(),
            "completed": completed,
            "episodes": len(episode_summaries),
            "episode_summaries": episode_summaries,
            "trace": trace,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.probe:
        probe(args.package)
    else:
        run_trace(args)


if __name__ == "__main__":
    main()
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_runner(
    package: str,
    tmp_path: Path,
    *,
    probe: bool = False,
    episodes: int = MAX_EPISODES,
    steps: int = MAX_STEPS_PER_EPISODE,
) -> dict:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    if package == "env-stableretro-turbo":
        cwd = repo_root
        pythonpath = str(repo_root)
        python_executable = os.environ.get("ENV_STABLERETRO_TURBO_PYTHON", sys.executable)
    else:
        cwd = tmp_path
        pythonpath = os.environ.get("STABLE_RETRO_ORACLE_PATH", "")
        python_executable = os.environ.get("STABLE_RETRO_ORACLE_PYTHON", sys.executable)

    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    else:
        env.pop("PYTHONPATH", None)

    command = [
        python_executable,
        "-c",
        RUNNER,
        "--package",
        package,
        "--steps",
        str(steps),
        "--episodes",
        str(episodes),
        "--repo-id",
        HF_LEVEL1_POLICY,
        "--filename",
        HF_LEVEL1_POLICY_FILENAME,
        "--cache-dir",
        str(repo_root / ".cache" / "hf_policy"),
    ]
    if probe:
        command.append("--probe")

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    stdout = completed.stdout.strip().splitlines()
    payload = json.loads(stdout[-1]) if stdout else {}
    if completed.returncode != 0:
        raise AssertionError(
            "policy parity subprocess failed\n"
            f"package={package}\n"
            f"returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}",
        )
    return payload


def _oracle_is_required() -> bool:
    return os.environ.get("STABLE_RETRO_REQUIRE_ORACLE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _skip_or_fail(reason: str) -> None:
    if _oracle_is_required():
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.mark.retro_oracle
def test_huggingface_level1_policy_trace_matches_stable_retro(tmp_path):
    stable_retro_probe = _run_runner("stable-retro", tmp_path, probe=True)
    if stable_retro_probe.get("status") != "ok":
        _skip_or_fail(stable_retro_probe.get("reason", "stable-retro oracle is unavailable"))

    turbo_probe = _run_runner("env-stableretro-turbo", tmp_path, probe=True)
    assert turbo_probe.get("status") == "ok", turbo_probe

    stable_retro_trace = _run_runner("stable-retro", tmp_path)
    if stable_retro_trace.get("status") != "ok":
        _skip_or_fail(stable_retro_trace.get("reason", "stable-retro policy trace is unavailable"))

    turbo_trace = _run_runner("env-stableretro-turbo", tmp_path)
    if turbo_trace.get("status") != "ok":
        _skip_or_fail(turbo_trace.get("reason", "env-stableretro-turbo policy trace is unavailable"))

    assert turbo_trace["completed"] is True, json.dumps(
        {
            "env_stableretro_turbo": {
                "version": turbo_trace.get("version"),
                "episode_summaries": turbo_trace.get("episode_summaries"),
            },
            "stable_retro": {
                "version": stable_retro_trace.get("version"),
                "episode_summaries": stable_retro_trace.get("episode_summaries"),
            },
        },
        indent=2,
        sort_keys=True,
    )
    assert stable_retro_trace["completed"] is True, json.dumps(
        stable_retro_trace.get("episode_summaries"),
        indent=2,
        sort_keys=True,
    )
    assert turbo_trace["trace"] == stable_retro_trace["trace"], json.dumps(
        {
            "env_stableretro_turbo": {
                "version": turbo_trace.get("version"),
                "completed": turbo_trace.get("completed"),
                "episode_summaries": turbo_trace.get("episode_summaries"),
                "tail": turbo_trace.get("trace", [])[-3:],
            },
            "stable_retro": {
                "version": stable_retro_trace.get("version"),
                "completed": stable_retro_trace.get("completed"),
                "episode_summaries": stable_retro_trace.get("episode_summaries"),
                "tail": stable_retro_trace.get("trace", [])[-3:],
            },
        },
        indent=2,
        sort_keys=True,
    )

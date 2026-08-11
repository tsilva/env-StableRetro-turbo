#!/usr/bin/env python
"""Play a Hugging Face SB3 policy with stable-retro-turbo."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from stable_retro.testing.hf_policy import (
    MARIO_SIMPLE_ACTIONS,
    _event_payload,
    _single_vector_info,
    load_sb3_policy,
    make_mario_level1_policy_env,
    resolve_hf_policy_path,
)


DEFAULT_REPO_ID = "tsilva/SuperMarioBros-NES_Level1"
DEFAULT_FILENAME = "ppo_supermariobros-nes-v0_4500000_steps.zip"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play the Super Mario Bros Level1-1 Hugging Face SB3 policy.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="Local SB3 .zip checkpoint. If omitted, download from Hugging Face.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument(
        "--event",
        default="level_change",
        choices=["level_change", "life_loss"],
        help="Scenario event to terminate on and report.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=672)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic argmax actions instead of stochastic policy sampling.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run policy steps without opening the playback window.",
    )
    return parser


def _policy_path(args: argparse.Namespace) -> Path:
    if args.policy is not None:
        return args.policy
    return resolve_hf_policy_path(
        args.repo_id,
        args.filename,
        env_var="STABLE_RETRO_HF_POLICY_PATH",
    )


def run(args: argparse.Namespace) -> int:
    model = load_sb3_policy(_policy_path(args), device=args.device)
    previous_render_skip = os.environ.get("STABLE_RETRO_DISABLE_RENDER_SKIP")
    if not args.no_window:
        os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = "1"
    env = make_mario_level1_policy_env()
    viewer = None
    if not args.no_window:
        from stable_retro.rendering import SimpleImageViewer

        viewer = SimpleImageViewer(maxwidth=args.max_width, scale_up=True)

    frame_delay = 0 if args.fps <= 0 else 1.0 / args.fps
    try:
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs, reset_infos = env.reset()
            previous_info = _single_vector_info(reset_infos, 0)
            for step in range(1, args.max_steps + 1):
                started_at = time.monotonic()
                if viewer is not None:
                    viewer.imshow(env.render())
                    if not viewer.isopen:
                        return 0

                action, _state = model.predict(
                    obs,
                    deterministic=args.deterministic,
                )
                action_value = int(np.asarray(action).reshape(-1)[0])
                masks = MARIO_SIMPLE_ACTIONS[
                    np.asarray(action, dtype=np.int64).reshape(-1)
                ]
                obs, rewards, terminations, truncations, infos = env.step(masks)

                info = _single_vector_info(infos, 0)
                payload = _event_payload(args.event, previous_info, info)
                previous_info = info
                if payload is not None:
                    print(
                        f"{args.event} episode={episode} step={step} "
                        f"action={action_value} reward={float(rewards[0]):g} "
                        f"payload={payload}",
                    )
                    return 0
                if bool(terminations[0] or truncations[0]):
                    break

                remaining = frame_delay - (time.monotonic() - started_at)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        if previous_render_skip is None:
            os.environ.pop("STABLE_RETRO_DISABLE_RENDER_SKIP", None)
        else:
            os.environ["STABLE_RETRO_DISABLE_RENDER_SKIP"] = previous_render_skip

    print(
        f"{args.event} did not fire within {args.episodes} episodes "
        f"and {args.max_steps} steps per episode",
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

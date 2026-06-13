#!/usr/bin/env python3
"""Record a side-by-side video of a native vector env frame stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = ROOT / "scripts" / "benchmark_vec_env.json"


def _load_profile(path: Path, name: str) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for profile in raw.get("profiles", []):
        if profile.get("name") == name:
            return profile
    available = ", ".join(sorted(p.get("name", "<unnamed>") for p in raw.get("profiles", [])))
    raise SystemExit(f"Unknown profile {name!r}. Available profiles: {available}")


def _parse_crop(value):
    if value is None:
        return None
    crop = tuple(int(part) for part in str(value).split(","))
    if len(crop) != 4:
        raise SystemExit("obs_crop must be top,bottom,left,right")
    return crop


def _sample_actions(env, rng: np.random.Generator) -> np.ndarray:
    # Use the env's action-space sampler, but seed it deterministically so the
    # recording is reproducible enough for before/after comparisons.
    return np.asarray([env.action_space.sample() for _ in range(env.num_envs)])


def _stack_canvas(obs: np.ndarray, frame_idx: int, scale: int, footer: str) -> Image.Image:
    if obs.ndim != 3 or obs.shape[-1] != 4:
        raise ValueError(f"expected one stacked grayscale obs with shape HxWx4, got {obs.shape}")

    height, width, channels = obs.shape
    panel_w = width * scale
    panel_h = height * scale
    top = 28
    bottom = 28
    canvas = Image.new("RGB", (panel_w * channels, panel_h + top + bottom), (14, 14, 14))
    draw = ImageDraw.Draw(canvas)
    labels = ("oldest", "t-2", "t-1", "newest")
    for channel in range(channels):
        frame = Image.fromarray(obs[:, :, channel], mode="L")
        frame = frame.resize((panel_w, panel_h), Image.Resampling.NEAREST).convert("RGB")
        x = channel * panel_w
        canvas.paste(frame, (x, top))
        draw.text((x + 8, 8), f"{labels[channel]} c{channel}", fill=(235, 235, 235))
    draw.text(
        (8, top + panel_h + 8),
        f"{footer} | frame {frame_idx}",
        fill=(220, 220, 220),
    )
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-json", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--profile", default="supermario-level1-1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")

    profile = _load_profile(Path(args.profiles_json), args.profile)

    import stable_retro as retro
    from stable_retro.vec_env import StableRetroNativeVecEnv

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-stable-retro")
    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"

    resize_h, resize_w = (int(part) for part in str(profile["resize"]).lower().split("x", 1))
    crop = _parse_crop(profile.get("obs_crop"))
    env = StableRetroNativeVecEnv(
        profile["game"],
        1,
        state=profile["state"],
        num_threads=1,
        copy_observations=False,
        render_mode="rgb_array",
        obs_resize=(resize_h, resize_w),
        obs_grayscale=bool(profile["grayscale"]),
        obs_crop=crop,
        obs_resize_algorithm=str(profile.get("resize_algorithm", "area")),
        frame_skip=int(profile["frame_skip"]),
        frame_stack=int(profile["frame_stack"]),
        maxpool_last_two=bool(profile.get("maxpool_last_two", True)),
    )
    env.action_space.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output = (
        Path(args.output)
        if args.output is not None
        else ROOT / "artifacts" / f"{args.profile}_standard_framestack.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    first_png = output.with_name(output.stem + "_first.png")
    footer = (
        f"{profile['game']} {profile['state']} | {profile['resize']} "
        f"{'gray' if profile['grayscale'] else 'rgb'} | crop {crop} | "
        f"{profile.get('resize_algorithm', 'area')} | skip {profile['frame_skip']} | "
        f"stack {profile['frame_stack']}"
    )

    try:
        obs = env.reset()[0]
        with tempfile.TemporaryDirectory(prefix="stable-retro-stack-frames-") as tmpdir:
            tmp = Path(tmpdir)
            for frame_idx in range(args.frames):
                canvas = _stack_canvas(obs, frame_idx, args.scale, footer)
                frame_path = tmp / f"frame_{frame_idx:05d}.png"
                canvas.save(frame_path)
                if frame_idx == 0:
                    canvas.save(first_png)
                actions = _sample_actions(env, rng)
                obs = env.step(actions)[0][0]

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    str(tmp / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
    finally:
        env.close()

    print(f"wrote {output}")
    print(f"wrote {first_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

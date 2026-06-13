<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="512" />

  **Fast Python 3.14 wheels for stable-retro RL workloads**
</div>

`stable-retro-turbo` publishes installable macOS Apple Silicon and Linux wheels for the upstream [`stable-retro`](https://github.com/Farama-Foundation/stable-retro) API surface.

Use it when you want `stable_retro` game environments without building the package and bundled public libretro cores from source yourself.

## What changed from upstream

This fork keeps the upstream `stable_retro` API and adds a small set of
RL-throughput features:

- Python `3.14` wheels for macOS `arm64` and Linux `x86_64`.
- Bundled Game Boy, NES, SNES, and Genesis/Master System public cores.
- Worker-local crop, resize, grayscale, frame skip, frame stack, max-pool,
  no-op reset, sticky actions, and reward clipping.
- Native C++ screen processing and fused `step_repeat_and_process()`.
- `StableRetroSubprocVecEnv` shared-memory observations to reduce IPC copying.
- `STABLE_RETRO_DISABLE_AUDIO=1` for RGB-only agents.
- `scripts/benchmark_vec_env.py` for baseline versus optimized throughput runs.

In local Mario benchmarks using `SuperMarioBros-Nes-v0`, the optimized path was
materially faster than the baseline Python/SB3 vector setup. The best measured
direct-ROM 8-env run was about `9,756 steps/s` versus about `4,076 steps/s` for
the baseline, roughly `2.39x` faster. Broader sweeps showed fused native
preprocessing helping most as env count rises, while true multi-env-per-process
batching is still blocked by stable-retro's current one-emulator-instance-per-process
native frontend.

## Install

```bash
python -m pip install stable-retro-turbo
```

Use it from Python:

```python
import stable_retro as retro

env = retro.make("Alleyway-GameBoy-v0", render_mode="rgb_array")
```

## RL preprocessing and SB3

For reinforcement-learning loops, image preprocessing can be done inside each
environment worker before observations are returned to the caller. This is useful
with `SubprocVecEnv`, where sending smaller observations across process
boundaries can be much faster than returning full-size RGB frames and resizing
later. This is the main speedup this fork adds over the upstream wrapper stack.

```python
import stable_retro as retro

env = retro.make(
    "SuperMarioBros-Nes-v0",
    render_mode="rgb_array",
    obs_resize=(84, 84),
    obs_resize_algorithm="nearest",  # nearest, bilinear, or area
    obs_grayscale=True,
)
```

Available image kwargs:

- `obs_resize=(height, width)`: resize image observations before they leave the env.
- `obs_resize_algorithm="nearest"`: choose `nearest`, `bilinear`, or `area`; `nearest` is fastest, while `area` is downscale-only and does more averaging work.
- `obs_grayscale=True`: return grayscale observations with shape `(height, width, 1)`.
- `obs_crop=(top, bottom, left, right)`: crop pixels before grayscale and resize.
- `frame_skip=4`: repeat each selected action inside the worker and sum rewards.
- `frame_stack=4`: stack recent observations inside the worker before IPC.
- `maxpool_last_two=True`: max-pool the last two skipped image frames.
- `noop_reset_max=30`: apply a random number of no-op reset steps.
- `sticky_action_prob=0.25`: probabilistically repeat the previous action.
- `reward_clip=True`: clip rewards to `[-1, 1]`.

Pass the same options through Stable-Baselines3 with `env_kwargs`:

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage


def make_mario_env(**kwargs):
    return retro.make(
        "SuperMarioBros-Nes-v0",
        render_mode="rgb_array",
        **kwargs,
    )


env = make_vec_env(
    make_mario_env,
    n_envs=8,
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={"start_method": "spawn"},
    env_kwargs={
        "obs_resize": (84, 84),
        "obs_resize_algorithm": "nearest",
        "obs_grayscale": True,
    },
)
env = VecTransposeImage(env)  # (n_envs, 1, 84, 84) for grayscale
```

For lower IPC overhead than `SubprocVecEnv`, use the shared-memory vector env:

```python
from stable_retro import StableRetroSubprocVecEnv

env = StableRetroSubprocVecEnv([make_mario_env for _ in range(8)])
```

The shared-memory vector env keeps observations in a parent-owned shared buffer,
so workers only send rewards, done flags, and infos through pipes on each step.
For Atari-style image rollouts this pairs well with env-local preprocessing and
fused native frame skipping:

```python
env = StableRetroSubprocVecEnv(
    [
        lambda: retro.make(
            "SuperMarioBros-Nes-v0",
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_grayscale=True,
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
        )
        for _ in range(16)
    ],
)
```

When possible, image preprocessing and repeated-step processing use native C++
helpers instead of Python image loops. The native path is selected automatically
for single-player image observations with no rotation or movie recording. Set
`STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS=1` or
`STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP=1` to force the Python fallback while
debugging or benchmarking.

`StableRetroChunkedSubprocVecEnv` is also available as an experimental generic
Gymnasium vector env that puts multiple envs in each worker process:

```python
from stable_retro import StableRetroChunkedSubprocVecEnv

env = StableRetroChunkedSubprocVecEnv(env_fns, chunk_size=4)
```

This is useful for envs that support multiple instances per process. Current
native stable-retro emulator instances do **not**: the C++ libretro frontend has
one active emulator/core callback target per process, so stable-retro games must
still use one emulator process per env. For stable-retro games, prefer
`StableRetroSubprocVecEnv` until the native frontend is refactored for true
multi-instance execution.

If your agent does not use audio, set `STABLE_RETRO_DISABLE_AUDIO=1` before
creating environments. This keeps RGB observations enabled while skipping audio
capture and supported core-side audio generation.

The deprecated compatibility import still works:

```python
import retro
```

For local development:

```bash
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
brew install cmake pkg-config lua@5.4 libzip
python -m pip install -U pip build cibuildwheel pytest pre-commit
python -m pip install -e .
```

## Commands

```bash
python -m pip install stable-retro-turbo          # install the published package
python -m pip install -e .                        # build and install this checkout
python -m build --wheel                           # build a local wheel
python -m cibuildwheel . --output-dir wheelhouse  # build release-style wheels
pytest                                            # run Python tests
pre-commit run --all-files                        # run repository hooks
cmake . && make -j2 && make -j2 -f tests/Makefile && ctest --progress --verbose
python scripts/benchmark_vec_env.py --game SuperMarioBros-Nes-v0 --num-envs 8
```

## Notes

- Published wheels target Apple Silicon `arm64` on macOS `14.0+` and `x86_64` on Linux, for Python `3.14`.
- Package versions follow the upstream `stable-retro` base version with this fork's patch number as a PEP 440 post-release suffix, for example `1.0.0.post1`.
- The public wheel build includes Game Boy, NES, SNES, and Sega Master System cores: `gambatte`, `fceumm`, `snes9x`, and `genesis_plus_gx`.
- CapnProto is disabled in the public wheel build path.
- SNES on Apple Silicon uses an automatic Rosetta helper because the native arm64 `snes9x` path is not stable across the bundled integrations.
- If Rosetta is not installed yet, install it once:

```bash
softwareupdate --install-rosetta --agree-to-license
```

- Release automation builds macOS arm64 and Linux x86_64 wheels, publishes them to PyPI, and attaches matching wheel files to GitHub Releases.
- See [`PUBLISHING.md`](PUBLISHING.md) for the release checklist.
- Upstream API and integration docs are still useful: [`docs/supported_emulators.md`](docs/supported_emulators.md), [`docs/supported_games.md`](docs/supported_games.md), and [`docs/macos_installation.md`](docs/macos_installation.md).

## Architecture

![stable-retro-turbo architecture diagram](./architecture.png)

## License

[MIT](LICENSE). Bundled third-party notices are listed in [`LICENSES.md`](LICENSES.md).

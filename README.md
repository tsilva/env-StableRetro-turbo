<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="512" />

  **Fast Python 3.14 wheels for stable-retro RL workloads**
</div>

`stable-retro-turbo` publishes installable macOS Apple Silicon and Linux wheels for the upstream [`stable-retro`](https://github.com/Farama-Foundation/stable-retro) API surface.

Use it when you want `stable_retro` game environments without building the package and bundled public libretro cores from source yourself.

## What changed from upstream

This fork keeps the upstream `stable_retro` API and focuses its optimized RL
surface on the native multithreaded vector path:

- Python `3.14` wheels for macOS `arm64` and Linux `x86_64`.
- Bundled Game Boy, NES, SNES, and Genesis/Master System public cores.
- Multi-emulator native frontend support inside one process, using per-instance
  libretro function tables, thread-local callback routing, and isolated core
  copies when several emulator instances load the same core.
- `StableRetroNativeVecEnv`, a same-process SB3 VecEnv where C++ owns the
  emulator pool, frame skip, preprocessing, frame stacking, autoreset, reward
  and done evaluation, and one contiguous batched observation buffer.
- Native crop, resize, grayscale, max-pool, no-op reset, sticky actions, reward
  clipping, and a persistent C++ worker pool for batched stepping.
- Native C++ screen processing and fused `step_repeat_and_process()` for the
  regular single-env API.
- `STABLE_RETRO_DISABLE_AUDIO=1` for RGB-only agents.
- `scripts/benchmark_vec_env.py` for native-vector throughput sweeps.

In local Mario benchmarks using `SuperMarioBros-Nes-v0`, the native vector path
is the supported and fastest measured runtime. It keeps the hot vector-env state
machine in C++ so rollout steps no longer cross Python once per environment.

Recent local direct-ROM fused-preprocessing results on
`SuperMarioBros-Nes-v0`:

| envs | native threads | native_vec_fused |
| ---: | -------------: | ---------------: |
| 8 | 8 | 7,797 steps/s |
| 16 | 16 | 8,295 steps/s |
| 32 | 32 | 8,632 steps/s |
| 32 | 16 | 8,910 steps/s |
| 64 | 16 | 7,662 steps/s |

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

For single environments, image preprocessing can still be done inside the env
before observations are returned to the caller:

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
- `frame_skip=4`: repeat each selected action inside the env and sum rewards.
- `frame_stack=4`: stack recent observations inside the env.
- `maxpool_last_two=True`: max-pool the last two skipped image frames.
- `noop_reset_max=30`: apply a random number of no-op reset steps.
- `sticky_action_prob=0.25`: probabilistically repeat the previous action.
- `reward_clip=True`: clip rewards to `[-1, 1]`.

For SB3-style Mario rollouts, use the native vector env directly:

```python
from stable_retro import StableRetroNativeVecEnv

env = StableRetroNativeVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=32,
    num_threads=16,
    render_mode="rgb_array",
    obs_resize=(84, 84),
    obs_grayscale=True,
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=True,
)
```

`StableRetroNativeVecEnv` currently targets homogeneous single-player image
rollouts with no movie recording and no screen rotation. It keeps the hot
rollout path in C++ and returns a contiguous NumPy observation batch shaped
`(num_envs, height, width, channels * frame_stack)`.

When possible, image preprocessing and repeated-step processing use native C++
helpers instead of Python image loops. The native path is selected automatically
for single-player image observations with no rotation or movie recording. Set
`STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS=1` or
`STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP=1` to force the Python fallback while
debugging or benchmarking.

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

<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="512" />

  **Fast Python 3.14 wheels for stable-retro RL workloads**
</div>

`stable-retro-turbo` is a Python package for classic-game reinforcement
learning environments. It keeps the upstream
[`stable-retro`](https://github.com/Farama-Foundation/stable-retro) API, ships
prebuilt public libretro cores, and adds a high-throughput native vector path
for SB3-style rollouts.

Use it when you want `stable_retro` environments with native frame skip,
cropping, resizing, grayscale conversion, frame stacking, reward/done handling,
and batched observations handled in C++.

## Install

```bash
python -m pip install stable-retro-turbo
```

Use it from Python:

```python
import stable_retro as retro

env = retro.make("Alleyway-GameBoy-v0", render_mode="rgb_array")
obs, info = env.reset()
```

For local development:

```bash
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
brew install cmake pkg-config lua@5.4 libzip
python -m pip install -U pip build cibuildwheel pytest pre-commit
python -m pip install -e .
```

## Native Vector Env

`StableRetroNativeVecEnv` is the supported fast path for homogeneous
single-player image rollouts. C++ owns the emulator pool, repeated stepping,
preprocessing, frame stack, autoreset, reward/done evaluation, and one
contiguous NumPy observation batch.

```python
from stable_retro import StableRetroNativeVecEnv

env = StableRetroNativeVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=32,
    num_threads=16,
    state="Level1-1",
    render_mode="rgb_array",
    obs_crop=(32, 0, 0, 0),
    obs_resize=(84, 84),
    obs_resize_algorithm="area",
    obs_grayscale=True,
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=True,
)
```

This returns observations shaped
`(num_envs, height, width, channels * frame_stack)`. The native vector env
currently targets image observations with one player, no movie recording, and
no screen rotation.

## Throughput

The latest local Mario benchmark uses the actual
`SuperMarioBros-Nes-v0` `Level1-1` state with the PPO-style preprocessing above:

```text
baseline:  4434.7 steps/s at 32 envs / 16 native threads
optimized: 9508.7 steps/s at 256 envs / 80 native threads
speedup:   2.14x
```

The main recent improvement is optional indexed-video preprocessing. NES frames
are internally 8-bit palette indices, so the NES core can expose those indices
directly. The native preprocessing path then uses small lookup tables for
grayscale and two-frame max-pool values instead of converting every sampled
pixel through a full RGB framebuffer first. Cores without indexed-video support
continue using the normal RGB framebuffer path.

Benchmark numbers are machine-dependent. Use actual game-state profiles for
training-representative measurements, and reserve `State.NONE` direct-ROM runs
for low-level emulator diagnostics.

## Commands

```bash
python -m pip install stable-retro-turbo          # install the published package
python -m pip install -e .                        # build and install this checkout
python -m build --wheel                           # build a local wheel
python -m cibuildwheel . --output-dir wheelhouse  # build release-style wheels
pytest                                            # run Python tests
pre-commit run --all-files                        # run repository hooks
cmake . && make -j2 && make -j2 -f tests/Makefile && ctest --progress --verbose
python scripts/benchmark_vec_env.py --profile supermario-level1-1
python scripts/record_frame_stack_video.py --profile supermario-level1-1
python scripts/record_frame_stack_video.py --profile supermario-level1-1 --no-preprocessing
```

## Notes

- Published wheels target Python `3.14` on macOS Apple Silicon `arm64` and
  Linux `x86_64`.
- Package versions follow the upstream `stable-retro` base version with this
  fork's patch number as a PEP 440 post-release suffix, for example
  `1.0.0.post4`.
- Public wheels include `gambatte`, `fceumm`, `snes9x`, and
  `genesis_plus_gx` for Game Boy, NES, SNES, Genesis, and Master System style
  workloads.
- ROM files are not bundled. Import legally obtained ROMs through the normal
  stable-retro data flow, or pass `--rom-path` to benchmark/recording scripts
  that support direct ROM paths.
- Set `STABLE_RETRO_DISABLE_AUDIO=1` before creating environments when your
  agent only needs image observations.
- Set `STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS=1` or
  `STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP=1` when comparing against Python
  fallback image-processing paths.
- `scripts/record_frame_stack_video.py` records the actual model-facing
  observation stack. Add `--no-preprocessing` to record the raw RGB screen from
  the same state for comparison.
- SNES on Apple Silicon uses an automatic Rosetta helper because the native
  arm64 `snes9x` path is not stable across the bundled integrations. If needed,
  install Rosetta once:

```bash
softwareupdate --install-rosetta --agree-to-license
```

- Release automation builds macOS arm64 and Linux x86_64 wheels, publishes them
  to PyPI, and attaches matching wheels to GitHub Releases. See
  [`PUBLISHING.md`](PUBLISHING.md).
- Upstream API and integration docs are still useful:
  [`docs/supported_emulators.md`](docs/supported_emulators.md),
  [`docs/supported_games.md`](docs/supported_games.md), and
  [`docs/macos_installation.md`](docs/macos_installation.md).

## Architecture

![stable-retro-turbo architecture diagram](./architecture.png)

## License

[MIT](LICENSE). Bundled third-party notices are listed in [`LICENSES.md`](LICENSES.md).

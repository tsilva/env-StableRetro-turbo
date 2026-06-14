<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="512" />

  **Fast Python 3.14 wheels for stable-retro RL workloads**
</div>

`stable-retro-turbo` is a performance fork of
[`stable-retro`](https://github.com/Farama-Foundation/stable-retro). Use it when
you want the upstream API, but need much faster image rollouts for reinforcement
learning.

The main fast path is `StableRetroNativeVecEnv`: C++ owns the emulator pool,
frame skip, image preprocessing, frame stacking, reward/done evaluation,
autoreset, and the batched NumPy observation buffer.

## Why Use This

- Native vector rollouts for homogeneous single-player image environments.
- Fused stepping and preprocessing, so Python is not looping over envs and
  frames.
- Native crop, resize, grayscale, frame skip, frame stack, and two-frame
  max-pool.
- Indexed-video preprocessing for NES cores that can expose palette indices,
  avoiding full RGB conversion before grayscale/resize.
- Prebuilt public cores in the wheels: `gambatte`, `fceumm`, `snes9x`, and
  `genesis_plus_gx`.

## Benchmark

Latest local benchmark: Super Mario Bros Level 1-1 with PPO-style Atari
preprocessing.

| Run | Computer | OS / arch | Emulated platform | Game / state | Envs | Threads | Preprocessing | Throughput |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| Baseline native vec | MacBook Pro, Apple M1 Pro, 8 cores, 16 GB RAM | macOS 26.5.1 / arm64 | NES via `fceumm` | `SuperMarioBros-Nes-v0` / `Level1-1` | 32 | 16 | crop `(32,0,0,0)`, resize `84x84` area, grayscale, frame skip `4`, frame stack `4`, max-pool last two frames, sampled actions, audio disabled | 4,434.7 steps/s |
| Optimized indexed-video native vec | MacBook Pro, Apple M1 Pro, 8 cores, 16 GB RAM | macOS 26.5.1 / arm64 | NES via `fceumm` | `SuperMarioBros-Nes-v0` / `Level1-1` | 256 | 80 | same settings | 9,508.7 steps/s |

Speedup: **2.14x** total rollout throughput in the best observed local
configuration above.

Re-run the default profile:

```bash
python3.14 scripts/benchmark_vec_env.py --profile supermario-level1-1
```

Override scale when comparing machines:

```bash
python3.14 scripts/benchmark_vec_env.py --profile supermario-level1-1 --num-envs 256 --num-threads 80
```

## Install

```bash
python3.14 -m pip install stable-retro-turbo
```

```python
import stable_retro as retro

env = retro.StableRetroNativeVecEnv(
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

## Development

```bash
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
brew install cmake pkg-config lua@5.4 libzip
python3.14 -m pip install -U pip build cibuildwheel pytest pre-commit
python3.14 -m pip install -e .
```

Useful commands:

```bash
pytest                                            # run Python tests
python3.14 -m build --wheel                           # build a local wheel
python3.14 -m cibuildwheel . --output-dir wheelhouse  # build release-style wheels
python3.14 scripts/benchmark_vec_env.py --list-profiles
python3.14 scripts/record_frame_stack_video.py --profile supermario-level1-1
```

## Notes

- Published wheels target Python `3.14` on macOS Apple Silicon `arm64` and
  Linux `x86_64`.
- ROM files are not bundled.
- Set `STABLE_RETRO_DISABLE_AUDIO=1` for image-only training benchmarks.
- Set `STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS=1` or
  `STABLE_RETRO_DISABLE_NATIVE_FUSED_STEP=1` when comparing against fallback
  paths.
- `StableRetroNativeVecEnv` currently targets image observations with one
  player, no movie recording, and no screen rotation.

## License

[MIT](LICENSE). Bundled third-party notices are listed in [`LICENSES.md`](LICENSES.md).

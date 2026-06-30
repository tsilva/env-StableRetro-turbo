<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/logo.png" alt="stable-retro-turbo" width="260" />

  <p>
    <a href="https://pypi.org/project/stable-retro-turbo/">
      <img src="https://img.shields.io/pypi/v/stable-retro-turbo.svg" alt="PyPI version" />
    </a>
  </p>

  **🚀 Blazing-fast Stable Retro fork with native vectorization and preprocessing 🚀**
</div>

`stable-retro-turbo` is a performance-focused fork of [Stable Retro](https://stable-retro.farama.org/) that accelerates rollouts by moving vectorization and preprocessing entirely into native code.
That is faster because the hot path avoids repeatedly bouncing between Python and the emulator for each environment step. Instead, many environments can be stepped and transformed in one native batch, reducing Python interpreter overhead, wrapper dispatch, memory copies, and per-frame preprocessing cost. The result is higher rollout throughput, especially when running many parallel environments.

## Install

```bash
uv venv --python 3.14
uv pip install stable-retro-turbo
```

## Use

```python
import stable_retro as retro  # Uses the upstream-compatible import name.

# RetroVecEnv is a stable-retro-turbo interface
env = retro.RetroVecEnv(
    # stable-retro params (all stable-retro params can still be used)
    "SuperMarioBros-Nes-v0",      # Stable Retro game integration.
    state="Level1-1",             # Saved game state to load in each lane.

    # stable-retro-turbo specific params
    num_envs=32,                  # Number of emulator lanes stepped together.
    num_threads=16,               # Native worker threads for those lanes.
    render_mode="rgb_array",      # Return frame arrays instead of opening a window.
    obs_crop=(32, 0, 0, 0),       # Crop 32 pixels from the top before resizing.
    obs_resize=(84, 84),          # Resize observations natively for RL input.
    obs_resize_algorithm="area",  # Area resize is a good downsampling default.
    obs_grayscale=True,           # Convert RGB frames to grayscale natively.
    obs_layout="chw",             # Return channel-first tensors for PyTorch/SB3.
    obs_copy="safe_view",         # Avoid extra copies while keeping observations safe.
    frame_skip=4,                 # Repeat each action for 4 emulator frames.
    frame_stack=4,                # Stack the last 4 processed frames.
    frame_maxpool=True,           # Max-pool recent frames to reduce flicker.
    reset_noops=0,                # Disable random no-op starts for this example.
    action_sticky_prob=0.0,       # Disable sticky actions for deterministic stepping.
    info_filter="terminal",       # Only return full info payloads at episode end.
)

obs = env.reset()  # Shape follows the native preprocessing choices above.
obs, rewards, dones, infos = env.step(
    [env.action_space.sample() for _ in range(32)]  # One action per env lane.
)
env.close()  # Release native emulator resources.
```

## RetroVecEnv Parameters

| Parameter | What it controls |
| --- | --- |
| `num_envs` | Number of emulator lanes in the vector environment. |
| `num_threads` | Native worker threads; defaults to `num_envs` when omitted. |
| `rom_path` | Explicit ROM path for direct-ROM tests or external integrations. |
| `obs_resize` | Native resize target as `(width, height)`. |
| `obs_crop` | Native crop before resize, using the same crop contract as `RetroEnv`. |
| `obs_grayscale` | Convert image observations to grayscale natively. |
| `obs_resize_algorithm` | Resize algorithm: `"nearest"`, `"bilinear"`, or `"area"`. |
| `obs_layout` | Observation layout: `"hwc"` or `"chw"`. |
| `obs_copy` | Observation ownership mode: `"copy"`, `"safe_view"`, or benchmark-only `"unsafe_view"`. |
| `frame_skip` | Repeat each action for this many emulator frames. |
| `frame_stack` | Stack this many processed frames in each returned observation. |
| `frame_maxpool` | Max-pool the last two skipped frames before preprocessing. |
| `reset_noops` | Apply up to this many random no-op frames after reset. |
| `action_sticky_prob` | Probability of repeating the previous lane action instead of the requested action. |
| `reward_clip` | Clip rewards with the same semantics as the single-env preprocessing path. |
| `info_filter` | Info payload filter: `"all"`, `"terminal"`, `"none"`, or `{"mode": ..., "keys": (...)}`. |
| `done_on` | General per-lane terminal rules keyed by info-variable `change`, `increase`, or `decrease`. |

## Commands

```bash
uv run python scripts/benchmark_vec_env.py --list-profiles                         # show saved benchmark profiles
uv run python scripts/benchmark_vec_env.py --profile supermario-level1-1 --dry-run # print resolved env benchmark config
uv run python scripts/benchmark_vec_env.py --profile supermario-level1-1           # run native/classic rollout benchmark
uv run python scripts/benchmark_sb3_ppo.py --profile supermario-level1-1 --dry-run # print resolved PPO benchmark config
uv run pytest tests/test_python/test_vec_env.py                                    # run focused RetroVecEnv tests
uv run --with build python -m build                                                # build source and wheel artifacts
```

## Benchmarks

The benchmark profile file is
[`scripts/benchmark_vec_env.json`](scripts/benchmark_vec_env.json). The default
user-facing profile is `supermario-level1-1`, which uses a real
`SuperMarioBros-Nes-v0` / `Level1-1` saved state with PPO-style preprocessing:
crop `(32,0,0,0)`, resize `84x84`, grayscale, frame skip `4`, frame stack `4`,
two-frame max-pool, `32` envs, and `16` native threads.

Use real saved states for user-facing throughput numbers. `State.NONE` is
reserved for explicit direct-ROM hot-path diagnostics and requires
`--allow-state-none`.

### Reference Modal Results

All rows were measured on Modal CPU with `cpu_request=16.0`,
`memory_mb=16384`, `os_cpu_count=32`, `affinity_cpu_count=32`,
`machine=x86_64`, platform `Linux-4.19.0-gvisor-x86_64-with-glibc2.36`, and
Python `3.14.6`. Each env-throughput row uses `SuperMarioBros-Nes-v0` /
`Level1-1`, a real saved state, crop `(32,0,0,0)`, resize `84x84`, grayscale,
frame skip `4`, frame stack `4`, two-frame max-pool, sampled actions, and
three samples.

The `stable-retro-turbo==1.0.0.post22` rows install the published PyPI wheel
and mount only the benchmark harness/profile JSON. Package runtime:
`/usr/local/lib/python3.14/site-packages/stable_retro/__init__.py`; extension
`/usr/local/lib/python3.14/site-packages/stable_retro/_retro.cpython-314-x86_64-linux-gnu.so`.
The upstream rows remotely build Farama `stable-retro` from
`ec7a62718a1f99f34bf5e5d5c57255c9a53df507` (`main`) and use the classic
`RetroEnv` path with benchmark-side preprocessing.

Modal runs: full env benchmark
[`ap-sNnRpf48gi4umUyabfrOA9`](https://modal.com/apps/eng-tiago-silva/main/ap-sNnRpf48gi4umUyabfrOA9);
16-env env-only fairness benchmark
[`ap-vQu3BlHG2uEeyykTbjK9pV`](https://modal.com/apps/eng-tiago-silva/main/ap-vQu3BlHG2uEeyykTbjK9pV).

| Source / backend | Shape | Samples steps/s | Mean | Std | Best | Speedup vs subproc / async | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| post22 native fused `RetroVecEnv` | `16` envs | `9403.0`, `9278.3`, `9612.2` | `9431.2` | `168.7` | `9612.2` | `10.13x` / `11.68x` | [post22 16-env][bench-post22-16env] |
| upstream Gymnasium `AsyncVectorEnv` / classic `RetroEnv` | `16` envs | `807.3`, `810.9`, `804.2` | `807.5` | `3.4` | `810.9` | `0.87x` / `1.00x` | [upstream async][bench-upstream-async] |

[bench-post22-16env]: artifacts/benchmarks/modal-1.0.0.post22-native-16env-2026-06-29.json
[bench-upstream-async]: artifacts/benchmarks/modal-upstream-stable-retro-async-16env-2026-06-29-2025.json

## Notes

- The import package is `stable_retro`; `retro` remains as a compatibility shim.
- `RetroVecEnv` requires `stable-baselines3` because it implements the SB3
  `VecEnv` interface.
- Source builds and CI cover Python `3.10` through `3.14`; the repo-local
  deterministic release helper currently targets Python `3.14` wheels.
  Building from source also requires CMake, a C/C++ compiler, and platform core
  build dependencies.
- ROMs are not included. Import ROMs and read game/core docs through upstream
  Stable Retro unless the work is specifically about this turbo layer.
- `done_on` terminates and autoresets only lanes whose configured
  info-variable rule fires. Supported ops are `change`, `increase`, and
  `decrease`.
- Use `done_on={"life_loss": ("lives", "decrease")}` for first-life-loss
  terminal transitions.
- `active_state_indices()` returns a read-only `int32` NumPy view for
  task-conditioned training; copy it when you need a stable snapshot.
- Third-party emulator cores carry their own licenses; see [`LICENSES.md`](LICENSES.md).

## Architecture

![stable-retro-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/architecture.png)

## License

[MIT](LICENSE)

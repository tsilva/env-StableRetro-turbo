<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/logo.png" alt="stable-retro-turbo" width="260" />

  <p>
    <a href="https://pypi.org/project/stable-retro-turbo/">
      <img src="https://img.shields.io/pypi/v/stable-retro-turbo.svg" alt="PyPI version" />
    </a>
  </p>

  **🚀 Blazing-fast Stable Retro fork with native vectorization and preprocessing 🚀**
</div>

stable-retro-turbo is a performance-focused Python package for
reinforcement-learning engineers who use Stable Retro environments and need
faster batched rollouts. It keeps the familiar `stable_retro` / `retro` import
surface, then adds `RetroVecEnv`, a native vector environment that moves
emulator stepping, preprocessing, reward/info handling, and autoreset work out
of Python.

This repository is only the turbo layer. For inherited Stable Retro material
such as the full game catalog, emulator support matrix, ROM import workflow,
integration UI, general tutorials, and community support, use the upstream
[Farama Stable Retro repository](https://github.com/Farama-Foundation/stable-retro)
and [Stable Retro docs](https://stable-retro.farama.org/).

## Install

```bash
uv venv --python 3.14
uv pip install stable-retro-turbo
```

For Stable Baselines3 training examples and local tests, add:

```bash
uv pip install stable-baselines3 pytest
```

Check the installed package:

```bash
uv run python - <<'PY'
import stable_retro as retro

print(retro.__version__.strip())
print(retro.RetroVecEnv)
PY
```

If you only need upstream Stable Retro behavior, install and read upstream
Stable Retro instead.

For development from source:

```bash
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
uv venv --python 3.14
uv pip install -e .
uv pip install stable-baselines3 pytest
```

## Use

```python
import stable_retro as retro

env = retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    state="Level1-1",
    num_envs=32,
    num_threads=16,
    render_mode="rgb_array",
    obs_crop=(32, 0, 0, 0),
    obs_resize=(84, 84),
    obs_resize_algorithm="area",
    obs_grayscale=True,
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=True,
    info_mode="terminal",
)

obs = env.reset()
obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(32)])
env.close()
```

`RetroVecEnv` keeps the leading constructor fields aligned with upstream
`RetroEnv`; turbo-specific controls are keyword-only additions after `*`.

## RetroVecEnv Parameters

```python
retro.RetroVecEnv(
    game,
    state=retro.State.DEFAULT,
    scenario=None,
    info=None,
    use_restricted_actions=retro.Actions.FILTERED,
    record=False,
    players=1,
    inttype=retro.data.Integrations.STABLE,
    obs_type=retro.Observations.IMAGE,
    render_mode="human",
    *,
    num_envs=1,                  # number of emulator lanes in the vector env
    num_threads=None,            # native worker threads; defaults to num_envs
    rom_path=None,               # explicit ROM path for direct-ROM or external use
    copy_observations=True,      # copy obs arrays; disable for fewer copies
    obs_resize=None,             # native resize target as (width, height)
    obs_crop=None,               # native crop before resize
    obs_grayscale=False,         # convert image observations to grayscale
    obs_resize_algorithm="nearest", # "nearest", "bilinear", or "area"
    frame_skip=1,                # repeat each action for this many frames
    frame_stack=1,               # stack this many processed frames
    maxpool_last_two=False,      # max-pool the last two skipped frames
    noop_reset_max=0,            # random no-op frames after reset
    sticky_action_prob=0.0,      # chance to repeat the previous lane action
    reward_clip=False,           # clip rewards in the native path
    info_mode="all",             # info payload mode: "all", "terminal", or "none"
    info_keys=None,              # optional info variable names to emit
    obs_layout="hwc",            # observation layout: "hwc" or "chw"
    done_on_info=None,           # info-variable terminal rules
    unsafe_zero_copy=False,      # benchmark-only single-buffer aliasing
)
```

Inherited fields keep their upstream Stable Retro meaning. The native vector
path currently supports `players=1`, image observations, and no movie
recording. `state` also accepts turbo-only multi-state forms: a sequence with
one state per env lane, or a `{state_name: weight}` mapping sampled on reset.

| Turbo parameter | What it controls |
| --- | --- |
| `num_envs` | Number of emulator lanes in the vector environment. |
| `num_threads` | Native worker threads; defaults to `num_envs` when omitted. |
| `rom_path` | Explicit ROM path for direct-ROM tests or external integrations. |
| `copy_observations` | Return copied observations; disable to reduce copies while keeping SB3-safe double buffering. |
| `obs_resize` | Native resize target as `(width, height)`. |
| `obs_crop` | Native crop before resize, using the same crop contract as `RetroEnv`. |
| `obs_grayscale` | Convert image observations to grayscale natively. |
| `obs_resize_algorithm` | Resize algorithm: `"nearest"`, `"bilinear"`, or `"area"`. |
| `frame_skip` | Repeat each action for this many emulator frames. |
| `frame_stack` | Stack this many processed frames in each returned observation. |
| `maxpool_last_two` | Max-pool the last two skipped frames before preprocessing. |
| `noop_reset_max` | Apply up to this many random no-op frames after reset. |
| `sticky_action_prob` | Probability of repeating the previous lane action instead of the requested action. |
| `reward_clip` | Clip rewards with the same semantics as the single-env preprocessing path. |
| `info_mode` | Info payload mode: `"all"`, `"terminal"`, or `"none"`. |
| `info_keys` | Optional sequence of info variable names to emit. |
| `obs_layout` | Observation layout: `"hwc"` or `"chw"`. |
| `done_on_info` | General per-lane terminal rules keyed by info-variable `change`, `increase`, or `decrease`. |
| `unsafe_zero_copy` | Benchmark-only single-buffer observation aliasing; requires `copy_observations=False`. |

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
- `done_on_info` terminates and autoresets only lanes whose configured
  info-variable rule fires. Supported ops are `change`, `increase`, and
  `decrease`.
- Use `done_on_info={"life_loss": ["lives", "decrease"]}` for first-life-loss
  terminal transitions.
- `active_state_indices()` returns a read-only `int32` NumPy view for
  task-conditioned training; copy it when you need a stable snapshot.
- Third-party emulator cores carry their own licenses; see [`LICENSES.md`](LICENSES.md).

## Architecture

![stable-retro-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/architecture.png)

## License

[MIT](LICENSE)

<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/logo.png" alt="stable-retro-turbo" width="260" />

  <p>
    <a href="https://pypi.org/project/stable-retro-turbo/">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fstable-retro-turbo%2Fjson&amp;query=%24.info.version&amp;label=pypi&amp;prefix=v&amp;cacheSeconds=300" alt="PyPI version" />
    </a>
  </p>

  **🚀 Blazing-fast Stable Retro fork with native vectorization and preprocessing 🚀**
</div>

`stable-retro-turbo` is a performance-focused fork of [Stable Retro](https://stable-retro.farama.org/) that accelerates rollouts by moving vectorization and preprocessing into native code. By batching environment stepping and preprocessing natively, it avoids Python overhead, wrapper dispatch, and redundant memory copies, delivering substantially higher rollout throughput, especially with many parallel environments.

## Turbo-Only Features

Compared with upstream Stable Retro's single-environment `RetroEnv` API,
`stable-retro-turbo` adds a training-oriented fast path:

- **⚡ RetroVecEnv**: `RetroVecEnv` provides a Gymnasium `VectorEnv`
  fast path for many emulator lanes at once.
- **🧪 Fused native preprocessing**: RL image transforms, frame handling, and
  reward clipping can run in the native rollout path.
- **📦 Observation ownership modes**: Callers can choose copy-safe observations
  or faster view-based observations for benchmarks.
- **🔎 Native info filtering**: Step metadata can be reduced before it crosses
  the Python boundary.
- **🛤️ Multi-state training support**: Lanes can use fixed, sampled, and tracked
  start states for curricula or task-conditioned agents.
- **🔁 Lane-local autoreset**: Finished lanes reset independently while keeping
  Gymnasium same-step `final_obs` / `final_info` metadata.
- **🚦 Info-transition terminals**: Episodes can end on changes in game info,
  such as first life loss.
- **🧾 Scenario-defined events**: Integrations can declare reusable named
  events in `scenario.json`, including verbose multi-trigger events such as
  `"op": "decrease"` or `"op": "change"`.
- **🎛️ Rollout stochasticity controls**: Sticky actions and random no-op starts
  are available in the native path.
- **🧩 Explicit ROM paths**: Benchmarks and integrations can run without relying
  on an imported ROM lookup.
- **🍄 Expanded Mario saved states**: Mario Level 1 and Level 2 starts ship with
  the package.
- **🍄 Super Mario event hooks**: `SuperMarioBros-Nes-v0` includes built-in
  `life_loss` and `level_change` scenario events for policy evaluation and
  curriculum terminals.

## Install

```bash
uv venv --python 3.14
uv pip install stable-retro-turbo
```

## Quick platform player

For a local editable install, run:

```bash
uv tool install . -e
```

Then launch a representative installed game for a platform:

```bash
stable-retro-turbo play genesis
stable-retro-turbo play nes
stable-retro-turbo play SuperMarioBros-Nes-v0
stable-retro-turbo play all
```

To see the raw RGB game and the exact PPO-style preprocessed observation at
the same time, add `--show-obs`:

```bash
stable-retro-turbo play SuperMarioBros-Nes-v0 --show-obs
```

The second window tiles the four grayscale 84x84 frames that make up the
observation, after the standard top crop and area resize. It samples every
fourth frame while the RGB game continues rendering at its normal 60 FPS.

`all` opens one game per platform, sequentially. Use
`stable-retro-turbo play --list` to see the current ROM-backed platform choices.

## Use

```python
import stable_retro as retro  # Uses the upstream-compatible import name.

# RetroVecEnv is the stable-retro-turbo native vector interface.
env = retro.RetroVecEnv(
    # stable-retro params (all stable-retro params can still be used)
    "SuperMarioBros-Nes-v0",      # Stable Retro game integration.
    state="Level1-1",             # Saved game state to load in each lane.

    # stable-retro-turbo specific params
    num_envs=32,                  # Number of emulator lanes stepped together.
    num_threads=16,               # Native worker threads for those lanes.
    render_mode="rgb_array",      # Return frame arrays instead of opening a window.
    obs_crop=(32, 0, 0, 0),       # Crop 32 pixels from the top before resizing.
    obs_crop_mode="mask",         # Hide the HUD while preserving full-frame geometry.
    obs_crop_fill=0,              # Pixel value used by mask crop regions.
    obs_resize=(84, 84),          # Resize observations natively for RL input.
    obs_resize_algorithm="area",  # Area resize is a good downsampling default.
    obs_grayscale=True,           # Convert RGB frames to grayscale natively.
    obs_layout="chw",             # Return channel-first tensors for PyTorch.
    obs_copy="safe_view",         # Avoid extra copies while keeping observations safe.
    frame_skip=4,                 # Repeat each action for 4 emulator frames.
    frame_stack=4,                # Stack the last 4 processed frames.
    maxpool_last_two=True,        # Max-pool recent frames to reduce flicker.
    noop_reset_max=0,             # Disable random no-op starts for this example.
    sticky_action_prob=0.0,       # Disable sticky actions for deterministic stepping.
    info_filter="terminal",       # Only return full info payloads at episode end.
)

obs, infos = env.reset(seed=123)
obs, rewards, terminations, truncations, infos = env.step(env.action_space.sample())
env.close()  # Release native emulator resources.
```

### Atari shared-core backend

Atari support is included in the standard install. Use `AtariVecEnv` for Atari
training runs:

```bash
pip install stable-retro-turbo
```

```python
import numpy as np
import stable_retro as retro

env = retro.AtariVecEnv(
    "Breakout-Atari2600-v0",
    state=retro.State.NONE,
    num_envs=32,
    num_threads=16,
    noop_reset_max=0,
    use_fire_reset=False,
    reward_clip=False,
)
obs, infos = env.reset(seed=123)
obs, rewards, terminations, truncations, infos = env.step(
    np.zeros(32, dtype=np.int64),
)
env.close()
```

`AtariVecEnv` is the sole supported Atari backend. It shares reentrant emulator
code across lanes and uses native discrete Atari actions. The old libretro/Stella
backend and its Stable Retro `.state` and button-mask contracts are unsupported.

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
    obs_resize=None,             # native resize target as (width, height)
    obs_crop=None,               # native crop before resize
    obs_crop_mode="remove",      # "remove" crops pixels; "mask" fills them
    obs_crop_fill=0,             # fill value for obs_crop_mode="mask"
    obs_grayscale=False,         # convert image observations to grayscale
    obs_resize_algorithm="nearest", # "nearest", "bilinear", or "area"
    obs_layout="hwc",            # observation layout: "hwc" or "chw"
    obs_copy="copy",             # "copy", "safe_view", or "unsafe_view"
    frame_skip=1,                # repeat each action for this many frames
    frame_stack=1,               # stack this many processed frames
    maxpool_last_two=False,      # max-pool the last two skipped frames
    noop_reset_max=0,            # random no-op frames after reset
    sticky_action_prob=0.0,      # chance to repeat the previous lane action
    reward_clip=False,           # clip rewards in the native path
    info_filter="all",           # "all", "terminal", "none", or mode/keys mapping
    done_on=None,                # info-variable terminal rules
)
```

Inherited fields keep their upstream Stable Retro meaning. The native vector
path currently supports `players=1`, image observations, and no movie
recording. `state` also accepts turbo-only multi-state forms: a sequence with
one state per env lane, or a `{state_name: weight}` mapping sampled on reset.

| Turbo parameter | Default | What it controls |
| --- | --- | --- |
| `num_envs` | `1` | Number of emulator lanes in the vector environment. |
| `num_threads` | `None` (`num_envs`) | Native worker threads; when omitted, uses one worker per env lane. |
| `rom_path` | `None` | Explicit ROM path for direct-ROM tests or external integrations. |
| `obs_resize` | `None` | Native resize target as `(width, height)`. |
| `obs_crop` | `None` | Native crop before resize, using the same crop contract as `RetroEnv`. |
| `obs_crop_mode` | `"remove"` | Crop behavior: `"remove"` keeps the historical geometry-changing crop, while `"mask"` preserves the full observation canvas and fills the cropped regions before resize, layout conversion, and frame stacking. |
| `obs_crop_fill` | `0` | Scalar uint8 fill value for crop masking; RGB observations fill every channel with the same value. |
| `obs_grayscale` | `False` | Convert image observations to grayscale natively. |
| `obs_resize_algorithm` | `"nearest"` | Resize algorithm: `"nearest"`, `"bilinear"`, or `"area"`. |
| `obs_layout` | `"hwc"` | Observation layout: `"hwc"` or `"chw"`. |
| `obs_copy` | `"copy"` | Observation ownership mode: `"copy"`, `"safe_view"`, or benchmark-only `"unsafe_view"`. |
| `frame_skip` | `1` | Repeat each action for this many emulator frames. |
| `frame_stack` | `1` | Stack this many processed frames in each returned observation. |
| `maxpool_last_two` | `False` | Max-pool the last two skipped frames before preprocessing. |
| `noop_reset_max` | `0` | Apply up to this many random no-op frames after reset. |
| `sticky_action_prob` | `0.0` | Probability of repeating the previous lane action instead of the requested action. |
| `reward_clip` | `False` | Clip rewards with the same semantics as the single-env preprocessing path. |
| `info_filter` | `"all"` | Info payload filter: `"all"`, `"terminal"`, `"none"`, or `{"mode": ..., "keys": (...)}`. |
| `done_on` | `None` | General per-lane terminal rules keyed by info-variable `change`, `increase`, or `decrease`. |

Use crop masking when you want to hide HUDs or other static screen regions
during initial training without changing the spatial observation contract. For
example, `obs_crop=(32, 0, 0, 0), obs_crop_mode="mask"` hides the top 32 pixels
before resize and frame stacking, while keeping the same observation geometry as
the full-canvas path for later fine-tuning on unmasked observations.

## Gymnasium VectorEnv

`RetroVecEnv` is specific to `stable-retro-turbo` and implements the
Gymnasium vector API directly. It does not subclass SB3 `VecEnv`, and
`stable-baselines3` is not a runtime dependency for the native vector path.

The native env defaults to Gymnasium same-step autoreset semantics:

```python
obs, infos = env.reset(seed=seed)
obs, rewards, terminations, truncations, infos = env.step(actions)
```

When a lane terminates, the returned observation for that lane is already the
reset observation. The final episode observation and info are exposed through
`infos["final_obs"]` and `infos["final_info"]`. SB3 compatibility is a
downstream responsibility; for example, use an adapter in `rlab` rather than
expecting this package to emit SB3-only `terminal_observation`, `reset_infos`, or
`TimeLimit.truncated` fields.

For runtimes that own episode boundaries, select manual autoreset and reset only
finished lanes:

```python
import stable_retro
from gymnasium.vector import AutoresetMode

env = stable_retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=16,
    autoreset_mode=AutoresetMode.DISABLED,
)
obs, infos = env.reset(seed=list(range(16)))
obs, rewards, terminations, truncations, infos = env.step(actions)
done = terminations | truncations
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done})
```

In disabled mode, terminal observations and infos are returned directly from
`step()`. A terminated lane cannot be stepped again until it is selected by a
masked reset. Unselected lanes retain their emulator state, RNG stream,
observation/frame stack, and sticky-action history. Reset infos are columnar and
only mark selected lanes as present.

`reset_mask` must be a NumPy `bool` array with shape `(num_envs,)` and at least
one selected lane. A scalar reset seed expands to `seed + lane_index`; a seed
sequence must contain exactly one integer or `None` per lane. During a masked
reset, unselected seed entries are ignored. For environments constructed with a
start-state catalog, `start_indices` is an `int32` array of the same shape:

```python
mask = np.array([False, True, False, True], dtype=np.bool_)
starts = np.array([-1, 1, -1, 0], dtype=np.int32)
obs, reset_infos = env.reset(
    seed=[None, 101, None, 303],
    options={"reset_mask": mask, "start_indices": starts},
)
```

Nonnegative values select that catalog entry atomically; `-1` keeps the
configured fixed/per-lane/weighted policy. Values and seeds for unselected lanes
are ignored. `active_state_indices()` and `active_states()` change only for
selected lanes.

## Commands

```bash
uv run python scripts/benchmark_vec_env.py --list-profiles                         # show saved benchmark profiles
uv run python scripts/benchmark_vec_env.py --profile supermario-level1-1 --dry-run # print resolved env benchmark config
uv run python scripts/benchmark_vec_env.py --profile supermario-level1-1           # run native/classic rollout benchmark
uv run python scripts/benchmark_atari_alepy.py --dry-run                           # compare AtariVecEnv with direct ale-py
make benchmark-local BENCHMARK_ARGS=--dry-run GAME=MegaMan PLATFORM=Nes STATE=Level1
make benchmark GAME=SuperMarioBros PLATFORM=Nes STATE=Level1-1
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

Use real saved states for user-facing `RetroVecEnv` throughput numbers.
`State.NONE` is reserved for explicit direct-ROM hot-path diagnostics and
requires `--allow-state-none`. The Atari/ALE comparison uses the power-on-reset
AtariVecEnv contract exclusively.

`make benchmark` refreshes the local extension through `setup.py build_ext --inplace`
only when native/build inputs changed, then runs the same benchmark entrypoint
as `benchmark-local`. Set `GAME` and `PLATFORM` to compose a Stable Retro id
such as `MegaMan-Nes-v0`, or pass a full id through `GAME` and omit `PLATFORM`.
Set `STATE`, `BENCHMARK_PROFILE`, `BENCHMARK_SECONDS`, and `BENCHMARK_ARGS` to
override the profile defaults.

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
- `RetroVecEnv` is the turbo-specific Gymnasium vector environment.
- `AtariVecEnv` is the shared-core Atari vector environment and the only supported Atari backend.
- `RetroVectorEnv` is not exposed; `RetroVecEnv` is the public native vector API.
- Source builds and CI cover Python `3.11` through `3.14`; the repo-local
  deterministic release helper publishes `cp311`, `cp312`, `cp313`, and `cp314`
  wheels for the supported release platforms.
  Building from source also requires CMake, a C/C++ compiler, and platform core
  build dependencies.
- ROMs are not included. Import ROMs and read game/core docs through upstream
  Stable Retro unless the work is specifically about this turbo layer.
- `done_on` terminates and autoresets only lanes whose configured
  info-variable rule fires. Supported ops are `change`, `increase`, and
  `decrease`.
- Named `done_on` events are resolved from the selected scenario's `events`
  map, with legacy `metadata.json` `info_events` used only as a fallback.
  Events may use compact `(variables, op)` pairs or verbose `triggers`; multiple
  triggers for one event are OR'd together.
- Use `done_on=["life_loss"]` or
  `done_on={"life_loss": {"variables": "lives", "op": "decrease"}}` for
  first-life-loss terminal transitions.
- `active_state_indices()` returns a read-only `int32` NumPy view for
  task-conditioned training; copy it when you need a stable snapshot.
- `set_state_policy(...)` accepts the same string, sequence, or weighted mapping
  forms as constructor `state=` and updates the policy used by future
  resets/autoresets without interrupting active lanes.
- Third-party emulator cores carry their own licenses; see [`LICENSES.md`](LICENSES.md).

## Architecture

![stable-retro-turbo architecture diagram](https://raw.githubusercontent.com/tsilva/stable-retro-turbo/turbo/architecture.png)

## License

[MIT](LICENSE)

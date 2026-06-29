<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="260" />

  **🚀 Native vector rollouts for Stable Retro RL training 🚀**
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
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
uv venv --python 3.12
uv pip install -e .
uv pip install stable-baselines3 pytest
```

Check the local build from the repo root:

```bash
uv run python - <<'PY'
import stable_retro as retro

print(retro.__version__.strip())
print(retro.RetroVecEnv)
PY
```

If you only need upstream Stable Retro behavior, install and read upstream
Stable Retro instead. If you need a released turbo wheel, use this repository's
[GitHub release artifacts](https://github.com/tsilva/stable-retro-turbo/releases)
when available.

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
    num_envs=1,
    num_threads=None,
    rom_path=None,
    copy_observations=True,
    obs_resize=None,
    obs_crop=None,
    obs_grayscale=False,
    obs_resize_algorithm="nearest",
    frame_skip=1,
    frame_stack=1,
    maxpool_last_two=False,
    noop_reset_max=0,
    sticky_action_prob=0.0,
    reward_clip=False,
    info_mode="all",
    info_keys=None,
    obs_layout="hwc",
    terminate_on_life_loss=False,
    life_variable=None,
    done_on_info=None,
    unsafe_zero_copy=False,
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
| `terminate_on_life_loss` | Enable native first-life-loss terminal transitions. |
| `life_variable` | Info/data variable used with `terminate_on_life_loss`, such as `"lives"`. |
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

## Notes

- The import package is `stable_retro`; `retro` remains as a compatibility shim.
- `RetroVecEnv` requires `stable-baselines3` because it implements the SB3
  `VecEnv` interface.
- Source builds require Python `3.10` through `3.14`, CMake, a C/C++ compiler,
  and platform core build dependencies.
- ROMs are not included. Import ROMs and read game/core docs through upstream
  Stable Retro unless the work is specifically about this turbo layer.
- `done_on_info` terminates and autoresets only lanes whose configured
  info-variable rule fires. Supported ops are `change`, `increase`, and
  `decrease`.
- `terminate_on_life_loss=True, life_variable="lives"` is still accepted and
  compiles to the same native rule path as `done_on_info`.
- `active_state_indices()` returns a read-only `int32` NumPy view for
  task-conditioned training; copy it when you need a stable snapshot.
- Third-party emulator cores carry their own licenses; see [`LICENSES.md`](LICENSES.md).

## Architecture

![stable-retro-turbo architecture diagram](./architecture.png)

## License

[MIT](LICENSE)

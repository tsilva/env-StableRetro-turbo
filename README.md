<div align="center">
  <img src="./logo.png" alt="stable-retro-turbo" width="260" />

  **🚀 Blazing-fast Stable Retro fork with native vectorization and preprocessing 🚀**
</div>

`stable-retro-turbo` is a Python library for reinforcement-learning developers who need faster batched rollouts from classic console games. It keeps Stable Retro's game integrations and single-environment API, and adds `RetroVecEnv`, a Gymnasium vector environment that steps many libretro emulators and preprocesses observations in native code.

Install the package, import your legally obtained ROMs, and use the upstream-compatible `stable_retro` import. The native path is useful for parallel training workloads that would otherwise spend substantial time crossing between Python wrappers and individual emulator instances.

## Install

Release wheels require Python 3.14 and support macOS on Apple Silicon and Linux on x86-64.

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install stable-retro-turbo
python -m stable_retro.import /path/to/your/roms
```

ROMs are not included. The importer matches supported ROMs to Stable Retro's game integrations.

Check the games available on your machine, then open one in the interactive player:

```bash
stable-retro-turbo play --list
stable-retro-turbo play nes
```

Pass a full game ID such as `SuperMarioBros-Nes-v0`, or use `all` to open one imported game per platform. Add `--show-obs` to display the raw game beside its PPO-style preprocessed observation.

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
    obs_crop_mode="mask",
    obs_resize=(84, 84),
    obs_resize_algorithm="area",
    obs_grayscale=True,
    obs_layout="chw",
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=True,
    info_filter="terminal",
)

obs, infos = env.reset(seed=123)
obs, rewards, terminations, truncations, infos = env.step(
    env.action_space.sample()
)

done = terminations | truncations
if done.any():
    obs, reset_infos = env.reset(options={"reset_mask": done})

env.close()
```

`RetroVecEnv` uses Gymnasium's disabled-autoreset semantics. A finished lane keeps its terminal observation and cannot be stepped again until it is selected by a masked reset; unselected lanes keep their emulator state, RNG stream, frame stack, and sticky-action history.

The fast path also supports:

- native crop, mask, resize, grayscale, layout conversion, frame skip, frame stack, and two-frame max-pooling;
- fixed, per-lane, or weighted saved-state selection for curricula and task-conditioned training;
- copy-safe, safe-view, and benchmark-only unsafe-view observation ownership;
- sticky actions, random no-op starts, reward clipping, and native info filtering;
- Atari through the packaged Stella core, using the same `RetroEnv` and `RetroVecEnv` APIs.

The inherited `RetroEnv` API remains available for single-environment use. `RetroVecEnv` supports one player, image observations, and no movie recording.

## Develop

```bash
git clone https://github.com/tsilva/stable-retro-turbo.git
cd stable-retro-turbo
uv sync --frozen
```

Source builds require Python 3.14, CMake, a C/C++ compiler, and the platform dependencies needed by the selected emulator cores.

## Commands

Run these commands from the repository root:

```bash
uv run --frozen stable-retro-turbo play --list                                      # list imported games by platform
uv run --frozen python scripts/benchmark_vec_env.py --list-profiles                 # list benchmark profiles
uv run --frozen python scripts/benchmark_vec_env.py --profile supermario-level1-1  # benchmark native and classic paths
uv run --frozen --with pytest pytest tests/test_python/test_cli.py tests/test_python/test_benchmark_vec_env.py  # run quick tests
uv run --frozen --with build python -m build                                        # build source and wheel artifacts
```

Benchmark definitions live in [`scripts/benchmark_vec_env.json`](scripts/benchmark_vec_env.json). Use real saved states for representative throughput measurements; `State.NONE` is reserved for explicit direct-ROM diagnostics and requires `--allow-state-none`.

## Notes

- The distribution is `stable-retro-turbo`; the Python package is `stable_retro`. The legacy `retro` import remains as a compatibility shim.
- `RetroVecEnv` implements Gymnasium's vector API directly. It is not a Stable-Baselines3 `VecEnv`, and Stable-Baselines3 is not a runtime dependency.
- A scalar reset seed expands to `seed + lane_index`. Seed sequences must contain one integer or `None` per lane.
- Weighted state mappings and explicit per-lane state sequences can be passed through `state`; selected lanes can also be reset to catalog entries with `start_indices`.
- `active_state_indices()` returns a read-only NumPy view. Copy it when you need a stable snapshot.
- [Stable Retro](https://stable-retro.farama.org/) remains the source for inherited game, integration, and emulator documentation.
- Bundled emulator cores have their own licenses; see [`LICENSES.md`](LICENSES.md).

## Architecture

![stable-retro-turbo architecture diagram](./architecture.png)

## License

[MIT](LICENSE)

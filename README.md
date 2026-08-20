<div align="center">
  <img src="./logo.png" alt="env-StableRetro-turbo" width="260" />

  **🚀 Blazing-fast Stable Retro fork with native vectorization and preprocessing 🚀**
</div>

`env-StableRetro-turbo` is a Python library for reinforcement-learning developers who need faster batched rollouts from classic console games. It keeps Stable Retro's game integrations and single-environment API, and adds `RetroVecEnv`, a Gymnasium vector environment that steps many libretro emulators and preprocesses observations in native code.

Install the package, import your legally obtained ROMs, and use the upstream-compatible `env_stableretro_turbo` import. The native path is useful for parallel training workloads that would otherwise spend substantial time crossing between Python wrappers and individual emulator instances.

## Install

Release wheels require Python 3.14 and support macOS on Apple Silicon and Linux on x86-64.

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install env-stableretro-turbo
python -m env_stableretro_turbo.import /path/to/your/roms
```

ROMs are not included. The importer matches supported ROMs to Stable Retro's game integrations.

Check the games available on your machine, then open one in the interactive player:

```bash
env-stableretro-turbo play --list
env-stableretro-turbo play nes
env-stableretro-turbo play SuperMarioBros-Nes-v0 --press START
env-stableretro-turbo play Breakout-Atari2600-v0 --mode 32 --difficulty A
```

Pass a full game ID such as `SuperMarioBros-Nes-v0`, or use `all` to open one imported game per platform. Add `--show-obs` to display the raw game beside its PPO-style preprocessed observation.

`--press BUTTON[:COUNT]` applies repeatable startup inputs using the selected
game's own button names. Atari games additionally accept `--mode N`, which
pulses the console SELECT switch `N` times before RESET, and `--difficulty A`
or `--difficulty B`, which sets both console difficulty switches. For example,
Breakout mode values `0`, `4`, `8`, …, `44` select its twelve one-player
cartridge variants. Use `--state none` to launch a game from its power-on state.

## Use

```python
import gymnasium as gym
import numpy as np

env = gym.make_vec(
    "env_stableretro_turbo:EnvStableRetroTurbo-v0",
    game="SuperMarioBros-Nes-v0",
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
```

The module-qualified ID imports the package and registers the factory. This ID
is vector-only and requires an explicit `game`; `RetroVecEnv` remains available
for direct use. Stable Retro's existing scalar `env_stableretro_turbo.make()` and
`RetroEnv` APIs are unchanged and are not registered under the Turbo ID.

`RetroVecEnv` uses Gymnasium's disabled-autoreset semantics. A finished lane keeps its terminal observation and cannot be stepped again until it is selected by a masked reset; unselected lanes keep their emulator state, RNG stream, frame stack, and sticky-action history.

## Turbo Vector API v2

`RetroVecEnv` implements the strict Turbo Vector API v2:

- `metadata["turbo_api_version"]` is `2`,
  `metadata["transition_transport"]` is `"numpy"`, and `metadata["render_modes"]`
  advertises `rgb_array`.
- Immutable `capabilities` and `signal_schema` declarations describe supported
  features and the dtype, shape, and reset/step availability of every signal.
- `buttons`, `action_mode`, `action_preset`, `action_table`,
  `action_meanings`, and `action_table_hash` expose the resolved action
  semantics without provider-specific probing.
- `state_catalog` is an immutable ordered tuple. Callers select reset states
  with an `int32` `state_indices` array and inspect the read-only active indices
  with `active_state_indices()`; state sampling and lane routing remain
  caller-owned.
- `observation_ownership` and `observation_buffer_depth` declare the exact
  lifetime of returned observations. Rendering is opt-in: with
  `render_mode="rgb_array"`, `render_lane(index)` renders one lane,
  `get_images()` renders all lanes, and `render()` renders lane zero. With the
  default `render_mode=None`, the first two methods return `None` and
  `get_images()` returns one `None` entry per lane.

When `env.supports_live_snapshots` is true, live positions can be captured
without advancing emulation and restored into any lane of the same environment:

```python
capture_mask = np.zeros(env.num_envs, dtype=np.bool_)
capture_mask[0] = True
captured = env.capture_snapshots(capture_mask)

restore_mask = np.zeros(env.num_envs, dtype=np.bool_)
restore_mask[3] = True
starts = [None] * env.num_envs
starts[3] = captured[0]
obs, infos = env.reset(
    options={"reset_mask": restore_mask, "snapshots": starts},
)
env.close()
```

Handles are reusable, session-local, and intentionally not pickleable. A
single masked reset can mix snapshot starts with ordinary `state_indices`;
`infos["start_source"]` is `int8`: `0` means an environment state and `1`
means a snapshot.
Scripted scenarios and cores that cannot serialize exact state report the
capability as unavailable.

The fast path also supports:

- native crop, mask, resize, grayscale, layout conversion, frame skip, frame stack, and two-frame max-pooling;
- ordered reset-state catalogs (including power-on as index zero when no saved
  state exists) with explicit per-lane `state_indices`; state sampling and
  curriculum routing stay with the caller;
- copy-safe, safe-view, and benchmark-only unsafe-view observation ownership;
- sticky actions, random no-op starts, reward clipping, and native info filtering;
- Atari through the packaged Stella core, using the same `RetroEnv` and `RetroVecEnv` APIs.

The inherited `RetroEnv` API remains available for single-environment use. `RetroVecEnv` supports one player, image observations, and no movie recording.

## Develop

```bash
git clone https://github.com/tsilva/env-StableRetro-turbo.git
cd env-StableRetro-turbo
uv sync --frozen
```

Source builds require Python 3.14, CMake, a C/C++ compiler, and the platform dependencies needed by the selected emulator cores.

## Commands

Run these commands from the repository root:

```bash
uv run --frozen env-stableretro-turbo play --list                                      # list imported games by platform
uv run --frozen --with pytest pytest tests/test_python/test_cli.py                   # run quick tests
uv run --frozen --with build python -m build                                        # build source and wheel artifacts
```

Core and semantic changes additionally use a pinned original-Stable-Retro
oracle for Super Mario Bros. and Breakout. The compared fields, reproducible
commands, and release receipt gate are documented in
[`docs/semantic_oracle.md`](docs/semantic_oracle.md).

## Benchmark

In an official correctness-gated
[TurboBench 1.0.0](https://pypi.org/project/turbobench-cli/1.0.0/) comparison on
the matched `supermario/canonical-v1` workload,
`env-stableretro-turbo==1.0.1.post37` measured
2.1248x to 2.2465x the throughput of original `stable-retro==1.0.1` at 1, 16,
and 32 environments. This is a workload-specific result, not a claim across all
games or emulator cores. See [`BENCHMARKS.md`](BENCHMARKS.md) for exact SPS,
paired confidence intervals, protocol, host details, provenance, and the
reproduction command. Install the benchmark CLI with:

```bash
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-12T00:00:00Z \
  turbobench-cli==1.0.0
```

## Notes

- The distribution is `env-stableretro-turbo`; the Python package is `env_stableretro_turbo`. The upstream-compatible `retro` import remains available for scalar integrations, while new code should import `env_stableretro_turbo`.
- `RetroVecEnv` implements Gymnasium's vector API directly. It is not a Stable-Baselines3 `VecEnv`, and Stable-Baselines3 is not a runtime dependency.
- A scalar reset seed expands to `seed + lane_index`. Seed sequences must contain one integer or `None` per lane.
- `state_catalog` preloads an ordered saved-state catalog. Select reset lanes with `reset_mask` and their exact catalog entries with `state_indices`; Turbo does not sample states.
- `capture_snapshots(mask)` returns lane-aligned live handles for exact
  same-instance continuation. The caller owns archive selection, eviction, and
  curriculum policy; `handle.nbytes` exposes approximate payload size.
- `active_state_indices()` returns a read-only NumPy view. Copy it when you need a stable snapshot.
- [Stable Retro](https://stable-retro.farama.org/) remains the source for inherited game, integration, and emulator documentation.
- Bundled emulator cores have their own licenses; see [`LICENSES.md`](LICENSES.md).

## Local credentials

Private local values declared in `.keyenv.toml` live in macOS Keychain. Run
`keyenv doctor` to verify them and launch credential-dependent commands with
`keyenv run -- <command>`. Python, Node, and their child processes receive the
values through their normal environment APIs. Keep only public or non-secret
configuration in dotenv files.

## Architecture

![env-StableRetro-turbo architecture diagram](./architecture.png)

## License

[MIT](LICENSE)

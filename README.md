[![Python](https://img.shields.io/pypi/pyversions/stable-retro.svg)](https://pypi.org/project/stable-retro/) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://gymnasium.farama.org/" target = "_blank">
    <img src="docs/_static/img/stable-retro-text.png" width="500px" />
</a></p>

A fork of [gym-retro](https://github.com/openai/retro) ('lets you turn classic video games into Gymnasium environments for reinforcement learning') with additional games, emulators and supported platforms. Since gym-retro is in maintenance now, you can instead submit PRs with new games or features here in stable-retro.

This repository tracks upstream Stable Retro while carrying experimental performance work for reinforcement-learning rollouts.

- [Supported emulators](docs/supported_emulators.md)
- [Supported games/envs](docs/supported_games.md)

## Emulated Systems

| System| Linux | Windows | Apple |
| --- | --- | --- | --- |
| Atari 2600 | ✓ | ✓ | ✓ |
| NES | ✓ | ✓ | ✓ |
| SNES| ✓ | ✓ | ✓ |
| Nintendo 64 | ✓† | ✓† | — |
| Nintendo DS | ✓ | ✓ | ✓ |
| Gameboy/Color | ✓ | ✓ | ✓* |
| Gameboy Advance| ✓ | ✓ | ✓ |
| Sega Genesis | ✓ | ✓ | ✓ |
| Sega Master System | ✓ | ✓ | ✓ |
| Sega CD | ✓ | ✓ | ✓ |
| Sega 32X | ✓ | ✓ | ✓ |
| Sega Saturn | ✓ | ✓ | ✓ |
| Sega Dreamcast | ✓‡ | — | — |
| PC Engine | ✓ | ✓ | ✓ |
| Arcade Machines | ✓ | ✓ | — |

\* On Apple Silicon (arm64), Gambatte (GB) is skipped by default in the CMake build.

† Built by default when BUILD_N64=ON and OpenGL headers are available. If headers are missing, the build skips the N64 core.

‡ Only available when hardware rendering is enabled (ENABLE_HW_RENDER=ON). Hardware rendering support is currently Linux-only in this project.

## Supported Games

Currently over 1000 games are integrated including:

| Category | Games |
| --- | --- |
| Platformers | Super Mario World, Sonic The Hedgehog 2, Mega Man 2, Castlevania IV |
| Fighters | Mortal Kombat Trilogy, Street Fighter II, Fatal Fury, King of Fighters '98 |
| Sports | NHL94, NBA Jam, Baseball Stars |
| Puzzle | Tetris, Columns |
| Shmups | 1943, Thunder Force IV, Gradius III, R-Type |
| BeatEmUps | Streets Of Rage, Double Dragon, TMNT 2: The Arcade Game, Golden Axe, Final Fight |
| Racing | Super Hang On, F-Zero, OutRun |
| RPGs (experimental) | Pokemon Red, Legend Of Zelda, Final Fantasy, Dragon Warrior |

> **Note:** If the game you want is not included but is supported by one of the systems in the list above, an integration tool is provided to help add new games.

## Performance Work In This Repository

The main fast path is `RetroVecEnv`: C++ owns the emulator pool,
frame skip, image preprocessing, frame stacking, reward/done evaluation,
autoreset, and the batched NumPy observation buffer.

- Native vector rollouts for homogeneous single-player image environments.
- Fused stepping and preprocessing, so Python is not looping over envs and frames.
- Native crop, resize, grayscale, frame skip, frame stack, and two-frame max-pool.
- Indexed-video preprocessing for NES cores that can expose palette indices, avoiding full RGB conversion before grayscale/resize.

Latest local benchmark: Super Mario Bros Level 1-1 with PPO-style Atari
preprocessing.

Latest benchmark comparison, updated from the standard benchmark protocol on
macOS 26.5.1 / arm64. Each row uses `SuperMarioBros-Nes-v0` / `Level1-1`,
`32` envs, sampled actions, crop `(32,0,0,0)`, resize `84x84` area,
grayscale, frame skip `4`, frame stack `4`, max-pool last two frames, and
audio disabled.

| Version / build | Backend | Samples (steps/s) | Mean steps/s | Std steps/s | Speedup vs `.post0` |
| --- | --- | ---: | ---: | ---: | ---: |
| current checkout (`VERSION.txt` `1.0.0.post16`) | `native_vec_fused` | `9639.4`, `10956.4`, `10830.4` | `10475.4` | `726.7` | `5.41x` |
| `1.0.0.post0` vanilla baseline | `subproc_vec_retro` | `1530.3`, `2110.1`, `2173.2` | `1937.9` | `354.4` | `1.00x` |

When running a new release benchmark, update the comparison table above with
the requested/current build first and the `.post0` baseline second.

| Run | Computer | OS / arch | Emulated platform | Game / state | Envs | Threads | Preprocessing | Throughput |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| Baseline native vec | MacBook Pro, Apple M1 Pro, 8 cores, 16 GB RAM | macOS 26.5.1 / arm64 | NES via `fceumm` | `SuperMarioBros-Nes-v0` / `Level1-1` | 32 | 16 | crop `(32,0,0,0)`, resize `84x84` area, grayscale, frame skip `4`, frame stack `4`, max-pool last two frames, sampled actions, audio disabled | 4,434.7 steps/s |
| Optimized indexed-video native vec | MacBook Pro, Apple M1 Pro, 8 cores, 16 GB RAM | macOS 26.5.1 / arm64 | NES via `fceumm` | `SuperMarioBros-Nes-v0` / `Level1-1` | 256 | 80 | same settings | 9,508.7 steps/s |

Speedup: **2.14x** total rollout throughput in the best observed local
configuration above.

Re-run the default profile:

```bash
python3 scripts/benchmark_vec_env.py --profile supermario-level1-1
```

The benchmarker defaults to `--backend auto`: current wheels use
`RetroVecEnv`, while vanilla `.post0` wheels fall back to
`SubprocVecEnv` over classic `RetroEnv` with the same profile-level preprocessing
applied by the benchmark script. Force a path with `--backend native`,
`--backend subproc`, or `--backend dummy`.

Override scale when comparing machines:

```bash
python3 scripts/benchmark_vec_env.py --profile supermario-level1-1 --num-envs 256 --num-threads 80
```

```python
import stable_retro as retro

env = retro.RetroVecEnv(
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

`RetroVecEnv` keeps the leading constructor fields aligned with upstream
`RetroEnv`, so existing Stable Retro call patterns keep their positional
meaning. Vector/native controls are keyword-only additions after `*`:

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

Original Stable Retro fields:

| Field | Meaning |
| --- | --- |
| `game` | Integration/game id, such as `"SuperMarioBros-Nes-v0"`. |
| `state` | Initial save-state. In addition to upstream string/enum values, `RetroVecEnv` accepts a per-env sequence or a `{state: weight}` sampling mapping. |
| `scenario` | Scenario JSON name or path. |
| `info` | Data/info JSON name or path. |
| `use_restricted_actions` | Action-space mode from `retro.Actions`. |
| `record` | Upstream movie-recording option. `RetroVecEnv` rejects recording because the native vector path does not write BK2 movies. |
| `players` | Player count. `RetroVecEnv` currently supports `players=1`. |
| `inttype` | Integration set, such as `retro.data.Integrations.STABLE`. |
| `obs_type` | Observation source. `RetroVecEnv` currently supports image observations. |
| `render_mode` | Upstream render mode field. Native vector rollouts normally use `"rgb_array"` for training. |

Turbo-only fields:

| Field | Meaning |
| --- | --- |
| `num_envs` | Number of emulator lanes in the vector environment. |
| `num_threads` | Worker threads for native stepping; defaults to `num_envs` and is clamped by the native layer. |
| `rom_path` | Explicit ROM path, useful for direct-ROM tests or external integrations. |
| `copy_observations` | Return copied observation arrays. Disable to reduce copies while keeping SB3-safe double buffering. |
| `obs_resize` | Resize observations to `(width, height)` in the native preprocessing path. |
| `obs_crop` | Crop observations before resize, using the same crop contract as `RetroEnv`. |
| `obs_grayscale` | Convert image observations to grayscale natively. |
| `obs_resize_algorithm` | Resize algorithm: `"nearest"`, `"bilinear"`, or `"area"`. |
| `frame_skip` | Repeat each action for this many emulator frames. |
| `frame_stack` | Stack this many processed frames in each returned observation. |
| `maxpool_last_two` | Max-pool the last two skipped frames before preprocessing, Atari-style. |
| `noop_reset_max` | Apply up to this many random no-op frames after reset. |
| `sticky_action_prob` | Probability of repeating the previous lane action instead of the requested action. |
| `reward_clip` | Clip rewards with the same semantics as the single-env preprocessing path. |
| `info_mode` | Info payload mode: `"all"`, `"terminal"`, or `"none"`. |
| `info_keys` | Optional list of info variables to include when `info_mode` emits info. |
| `obs_layout` | Observation layout: `"hwc"` or `"chw"`. |
| `terminate_on_life_loss` | Enable first-life-loss terminal transitions using `life_variable`. |
| `life_variable` | Info/data variable name used by `terminate_on_life_loss`, for example `"lives"`. |
| `done_on_info` | General per-lane terminal rules keyed by info-variable changes, increases, or decreases. |
| `unsafe_zero_copy` | Benchmark-only single-buffer observation aliasing; requires `copy_observations=False`. |

Mixed start-state training can stay on the native vector path by passing a list
or dict to `state`. A list means fixed per-env slot assignment and must have one
entry per env. A dict maps state names to positive finite sampling weights;
weights are normalized before sampling. Sampling happens independently for each
env on every reset, and reset infos include both `start_state` and `state`.

```python
env = retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=32,
    num_threads=16,
    state={"Level1-1": 1.0, "Level1-2": 1.0, "Level1-3": 1.0},
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

For task-conditioned PPO, use the integer state-index view instead of per-step
`info` dict strings:

```python
import numpy as np

state_names = env.initial_state_names
state_indices = env.active_state_indices()

obs = env.reset()
task_ids = state_indices.copy()
task_one_hot = np.eye(len(state_names), dtype=np.float32)[task_ids]

obs, rewards, dones, infos = env.step(actions)
task_ids = state_indices.copy()
```

`active_state_indices()` returns the same read-only `int32` NumPy view each
time. It is owned by the native vector env and mutates in place after `reset()`
and after any per-lane automatic reset before the next observation is returned.
Call `.copy()` when code needs a stable snapshot.

In fixed per-env slot mode, duplicated state labels share one task index:

```python
env = retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=4,
    state=["Level1-1", "Level1-2", "Level1-1", "Level1-2"],
)

env.initial_state_names      # ("Level1-1", "Level1-2")
env.active_state_indices()   # [0, 1, 0, 1]
```

Benchmark the same mixed-level profile with:

```bash
python3 scripts/benchmark_vec_env.py --profile supermario-level1-1 --backend native --states Level1-1,Level1-2,Level1-3 --state-probs 1,1,1
```

First-life-loss episode termination can be enabled in the native vector path
for games whose data files expose a suitable life counter. This is opt-in
because not every game has a valid `lives` variable, and similarly named
variables are not guaranteed to mean the same thing across games:

```python
env = retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=32,
    state="Level1-1",
    terminate_on_life_loss=True,
    life_variable="lives",
)
```

For more general per-lane terminal transitions, use `done_on_info`. Each rule
compares the current info values against that lane's post-reset baseline. The
supported ops are `change`, `increase`, and `decrease`; keys can be a string or
a sequence of strings:

```python
env = retro.RetroVecEnv(
    "SuperMarioBros-Nes-v0",
    num_envs=16,
    state={"Level1-1": 0.5, "Level1-2": 0.5},
    done_on_info={
        "life_loss": ["lives", "decrease"],
        "level_change": [["levelHi", "levelLo"], "change"],
    },
)
```

Terminal info includes only the rules that fired. The `keys`, `prev`, and
`next` fields are always lists, including single-key rules, so metric code can
handle single-key and multi-key transitions the same way:

```python
info["done_on_info"] == {
    "level_change": {
        "keys": ["levelHi", "levelLo"],
        "op": "change",
        "prev": [0, 0],
        "next": [0, 1],
    },
}
```

The legacy `terminate_on_life_loss=True, life_variable="lives"` arguments are
still accepted and are compiled into the same `life_loss` rule internally. If
both legacy life loss and an explicit `done_on_info["life_loss"]` are provided,
the explicit rule wins.

## Installation

Stable Retro supports Python 3.10 through 3.14.

```
pip3 install stable-retro
```

or if the above doesn't work for your platform:

```
pip3 install git+https://github.com/Farama-Foundation/stable-retro.git
```

If you plan to integrate new ROMs, states or emulator cores or plan to edit an existing env:

```
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
pip3 install -e .
```

For platform-specific instructions including building from source, optional core dependencies, and the Integration UI:
- [Linux Installation](docs/linux_installation.md) - Ubuntu/Debian dependencies, N64 and Dreamcast core setup, WSL2 guide
- [macOS Installation](docs/macos_installation.md) - Apple Silicon build instructions, Homebrew dependencies

## Example

'Nature CNN' model trained using PPO on Airstriker-Genesis env (rom already included in the repo)

```
sudo apt-get update
sudo apt-get install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg
```

You need to install a stable baselines 3 version that supports gymnasium

```
pip3 install git+https://github.com/Farama-Foundation/stable-retro.git
pip3 install stable_baselines3[extra]
```

Start training:

```
cd retro/examples
python3 ppo.py --game='Airstriker-Genesis-v0'
```

More advanced examples:
[https://github.com/MatPoliquin/stable-retro-scripts](https://github.com/MatPoliquin/stable-retro-scripts)

## Documentation & Tutorials

Documentation is available at [https://stable-retro.farama.org/](https://stable-retro.farama.org/) (work in progress)

See [LICENSES.md](https://github.com/Farama-Foundation/stable-retro/blob/master/LICENSES.md) for information on the licenses of the individual cores.

| Topic | Description |
| --- | --- |
| [Windows WSL2 Setup](https://www.youtube.com/watch?v=vPnJiUR21Og) | Step-by-step guide for setting up stable-retro on Windows 11 with WSL2 and Ubuntu 22.04 |
| [Game Integration Tool](https://www.youtube.com/playlist?list=PLmwlWbdWpZVvWqzOxu0jVBy-CaRpYha0t) | Playlist covering how to use the integration tool to add new games |
| [RetroArch + ML Models](https://www.youtube.com/watch?v=hkOcxJvJVjk) | Running a custom RetroArch build that supports overriding player input with trained models |

## ROMs and BIOS files

Each game integration has files listing memory locations for in-game variables, reward functions based on those variables, episode end conditions, savestates at the beginning of levels and a file containing hashes of ROMs that work with these files.

Please note that ROMs are not included and you must obtain them yourself. Most ROM hashes are sourced from their respective No-Intro SHA-1 sums.

Run this script in the roms folder you want to import. If the checksum matches it will import them in the related game folder in stable-retro.

```bash
python3 -m retro.import .
```

Some platforms like Sega Saturn and Dreamcast also need to be provided a BIOS.
 [List of BIOS names and checksums](docs/core_bios.md).

The following non-commercial Sega Genesis ROM is included with Stable Retro for testing purposes:
- [Airstriker](https://pdroms.de/genesis/airstriker-v1-50-genesis-game) by Electrokinesis

 [List of other included ROMs](docs/included_roms.md).

## Contributing & Support

[See CONTRIBUTING.md](https://github.com/Farama-Foundation/stable-retro/blob/master/CONTRIBUTING.md)

For any issues, suggestions, or discussions related to Stable-Retro, please use [GitHub Issues](https://github.com/Farama-Foundation/stable-retro/issues) or the Farama Foundation's [Discord](https://discord.gg/aPjhD5cf).

## Supported specs:

Platforms:
- Windows 10, 11 (via WSL2)
- macOS 10.13 (High Sierra), 10.14 (Mojave)
- Linux (manylinux1). Ubuntu 24.04 is recommended

Python:
- Python 3.10 through 3.14

CPU with `SSE3` or better

## Citation

```
@misc{stable-retro,
  author = {Poliquin, Mathieu},
  title = {Stable Retro, a maintained fork of OpenAI's gym-retro},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Farama-Foundation/stable-retro}},
}
```

## Papers

List of papers mentioning stable-retro. If you want your paper to be added here please open a github issue.

* [Exploration-Driven Generative Interactive Environments](https://arxiv.org/pdf/2504.02515)
* [Gymnasium: A Standardized Interface for Reinforcement Learning Environments](https://openreview.net/pdf?id=qPMLvJxtPK)
* [IPR-1: Interactive Physical Reasoner](https://arxiv.org/html/2511.15407v1)
* [SAFE-SMART: Safety Analysis and Formal Evaluation using STL Metrics for Autonomous RoboTs](https://arxiv.org/html/2511.17781v1)
* [General Modular Harness for LLM Agents in Multi-Turn Gaming Environments](https://arxiv.org/abs/2507.11633v1)
* [ReactiveGWM: Steering NPC in Reactive Game World Models](https://arxiv.org/pdf/2605.15256)
* [Dissecting Discrete Soft Actor-Critic: Limitations and Principled Alternatives](https://arxiv.org/pdf/2509.09838)

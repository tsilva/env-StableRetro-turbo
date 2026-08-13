---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/stable-retro-text.png
:alt: Stable-Retro Logo
```

```{project-heading}
Retro games for Reinforcement Learning
```

```{figure} _static/img/retro_games.png
   :alt: Stable-retro gif
   :width: 500
```

**Stable-Retro is a maintained fork of OpenAI’s Retro library.**

stable-retro lets you turn classic video games into Gymnasium environments for reinforcement learning. Supported plateforms includes Sega Master System/Genesis/CD/32X/Saturn/Dreamcast, NES, SNES, Nintendo 64/DS, Atari 2600, Arcade Machines and more

- {doc}`Supported emulators <supported_emulators>`
- {doc}`Supported games <supported_games>`

```{code-block} python
"""Run batched native stable-retro rollouts through Gymnasium VectorEnv."""

import gymnasium as gym
import stable_retro as retro


env = gym.make_vec(
    "stable_retro:StableRetro-Turbo-v0",
    game="Airstriker-Genesis-v0",
    num_envs=8,
    state=retro.State.DEFAULT,
    render_mode="rgb_array",
    obs_resize=(84, 84),
    obs_grayscale=True,
    obs_layout="chw",
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=True,
    sticky_action_prob=0.25,
    reward_clip=True,
)

obs, infos = env.reset(seed=123)
obs, rewards, terminations, truncations, infos = env.step(env.action_space.sample())
env.close()
```

The registered Turbo ID is vector-only and requires `game`. Direct
`RetroVecEnv` construction remains available, and the inherited scalar
`stable_retro.make()` and `RetroEnv` APIs are unchanged. `RetroVecEnv` is
Gymnasium-first; SB3 `VecEnv` adaptation belongs downstream, for example in an
`rlab` adapter.

```{toctree}
:hidden:
:caption: Introduction

getting_started.md
linux_installation.md
macos_installation.md
included_roms.md
core_bios.md
supported_emulators.md
supported_games.md
developing.md
integration.md
python.md
```

[//]: # (```{toctree})
[//]: # (:hidden:)
[//]: # (:caption: Environments)
[//]: # ()
[//]: # (```)

```{toctree}
:hidden:
:caption: Development

release_notes.md
semantic_oracle.md
Github <https://github.com/Farama-Foundation/stable-retro>
Contribute to the Docs <https://github.com/Farama-Foundation/stable-retro/blob/master/docs/README.md>
```
[//]: # (release_notes/index)

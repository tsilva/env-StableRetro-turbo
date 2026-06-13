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
"""Train an agent using PPO with the native stable-retro vector env."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

import stable_retro as retro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis-v0")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    venv = VecTransposeImage(
        retro.StableRetroNativeVecEnv(
            args.game,
            num_envs=args.num_envs,
            state=args.state,
            scenario=args.scenario,
            num_threads=args.num_threads,
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_grayscale=True,
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
            sticky_action_prob=0.25,
            reward_clip=True,
        ),
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
```

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
Github <https://github.com/Farama-Foundation/stable-retro>
Contribute to the Docs <https://github.com/Farama-Foundation/stable-retro/blob/master/docs/README.md>
```
[//]: # (release_notes/index)

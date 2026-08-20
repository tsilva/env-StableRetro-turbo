# Python API

## RetroEnv

The Python API consists primarily of {func}`env_stableretro_turbo.make`, {class}`env_stableretro_turbo.RetroEnv`, and a few enums.  The main function most users will want is {func}`env_stableretro_turbo.make`.

```{eval-rst}
.. autofunction:: env_stableretro_turbo.make
```

```{eval-rst}
.. autoclass:: env_stableretro_turbo.RetroEnv
```

If you want to specify either the default state named in the game integration's `metadata.json` or specify that you want to start from the initial power on state of the console, you can use the {class}`env_stableretro_turbo.State` enum:

```{eval-rst}
.. autoclass:: env_stableretro_turbo.State
   :members:
```

## Actions

There are a few possible action spaces included with {class}`env_stableretro_turbo.RetroEnv`:

```{eval-rst}
.. autoclass:: env_stableretro_turbo.Actions
   :members:
```

`use_restricted_actions` also accepts a named table from the game's
`metadata.json`, or an inline ordered table of console button labels. Both forms
produce an exact `Discrete` space:

```python
env = env_stableretro_turbo.make(
    game="SuperMarioBros-Nes-v0",
    use_restricted_actions="simple",
)

custom = env_stableretro_turbo.make(
    game="SuperMarioBros-Nes-v0",
    use_restricted_actions=[[], ["RIGHT"], ["RIGHT", "A"]],
)
```

The resolved contract is available through `action_preset`, `action_table`,
`action_meanings`, and `action_table_hash`. Multiplayer exact tables contain one
complete per-player action in each category; Stable Retro does not construct a
Cartesian product for them.

## Observations

The default observations are RGB images of the game, but you can view RAM values instead (often much smaller than the RGB images and also your agent can observe the game state more directly).  If you want variable values, any variables defined in `data.json` will appear in the `info` dict after each step.

```{eval-rst}
.. autoclass:: env_stableretro_turbo.Observations
   :members:
```

## Multiplayer Environments

A small number of games support multiplayer.  To use this feature, pass `players=<n>` to {class}`env_stableretro_turbo.RetroEnv`.  Here is an example random agent that controls both paddles in `Pong-Atari2600`:

```{literalinclude} ../env_stableretro_turbo/examples/trivial_random_agent_multiplayer.py
```

## Replay files

Stable Retro can create  [.bk2](http://tasvideos.org/Bizhawk/BK2Format.html) files which are recordings of an initial game state and a series of button presses.  Because the emulators are deterministic, you will see the same output each time you play back this file.  Because it only stores button presses, the file can be about 1000 times smaller than storing the full video.

In addition, if you wish to use the stored button presses for training, they may be useful.  For example, there are [replay files for each Sonic The Hedgehog level](https://github.com/openai/retro-movies) that were made available for the [Stable Retro Contest](https://openai.com/blog/retro-contest/).

You can create and view replay files using the {ref}`integration-ui` (Game > Play Movie...).  If you want to use replay files from Python, see the following sections.

### Record

If you have an agent playing a game, you can record the gameplay to a `.bk2` file for later processing:

```python
import env_stableretro_turbo

env = env_stableretro_turbo.make(game='Airstriker-Genesis-v0', record='.')
env.reset()
while True:
    _, _, terminate, truncate, _ = env.step(env.action_space.sample())
    if terminate or truncate:
        break
```

### Playback

Given a `.bk2` file you can load it in python and either play it back or use the actions for training.

```python
import env_stableretro_turbo

movie = env_stableretro_turbo.Movie('Airstriker-Genesis-Level1-000000.bk2')
movie.step()

env = env_stableretro_turbo.make(
    game=movie.get_game(),
    state=None,
    # bk2s can contain any button presses, so allow everything
    use_restricted_actions=env_stableretro_turbo.Actions.ALL,
    players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    env.step(keys)
```

### Render to Video

This requires [ffmpeg](https://www.ffmpeg.org/) to be installed and writes the output to the directory that the input file is located in.

```shell
python3 -m env_stableretro_turbo.scripts.playback_movie Airstriker-Genesis-Level1-000000.bk2
```

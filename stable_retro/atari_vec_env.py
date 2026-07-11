"""Owned native ALE vector environment with lane-local lifecycle control.

The emulator and vector scheduler are compiled into :mod:`stable_retro._retro`.
``ale-py`` remains the ROM registry, but its vector backend is not used here.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from stable_retro import _retro
from stable_retro.enums import Actions, State

try:
    from ale_py import roms
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without dependency
    raise ModuleNotFoundError(
        "AtariVecEnv requires ale-py for its ROM registry; "
        "install stable-retro-turbo[atari]",
    ) from exc


def ale_game_id(game: str) -> str:
    """Translate a Stable Retro Atari game id to ale-py's ROM id."""
    value = str(game).strip()
    if not value:
        raise ValueError("game must not be empty")
    if value.endswith("-v0"):
        value = value[:-3]
    if value.endswith("-Atari2600"):
        value = value[: -len("-Atari2600")]
    elif "-" in value:
        raise ValueError("AtariVecEnv only supports Atari2600 game ids")

    out: list[str] = []
    for index, char in enumerate(value):
        if char == "-":
            out.append("_")
        elif char.isupper() and index and value[index - 1].islower():
            out.extend(("_", char.lower()))
        else:
            out.append(char.lower())
    result = "".join(out).strip("_")
    if not result:
        raise ValueError("game must contain an Atari ROM name")
    return result


def _power_on_state(state) -> bool:
    if state is None or state is State.NONE:
        return True
    return str(state).strip().lower() in {"none", "state.none"}


def _normalize_autoreset_mode(value: AutoresetMode | str) -> AutoresetMode:
    if isinstance(value, AutoresetMode):
        return value
    try:
        return AutoresetMode(value)
    except ValueError:
        normalized = str(value).replace("_", "").replace("-", "").lower()
        aliases = {
            "disabled": AutoresetMode.DISABLED,
            "nextstep": AutoresetMode.NEXT_STEP,
            "samestep": AutoresetMode.SAME_STEP,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"Invalid autoreset_mode: {value}") from exc


class AtariVecEnv(VectorEnv):
    """High-throughput Atari vector environment with a native ALE backend.

    In ``DISABLED`` mode, a terminal lane remains terminal and cannot be
    stepped until it is selected by ``reset_mask``. Masked reset work is
    scheduled lane-locally in native code; unselected lanes are only read to
    produce the returned vector observation.
    """

    backend = "atari-v2"
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        game: str,
        state=State.NONE,
        use_restricted_actions=Actions.FILTERED,
        *,
        num_envs: int = 1,
        batch_size: int = 0,
        num_threads: int = 0,
        thread_affinity_offset: int = -1,
        max_episode_steps: int = 108_000,
        obs_resize=(84, 84),
        obs_grayscale: bool = True,
        frame_skip: int = 4,
        frame_stack: int = 4,
        maxpool_last_two: bool = True,
        noop_reset_max: int = 30,
        sticky_action_prob: float = 0.0,
        reward_clip: bool = True,
        use_fire_reset: bool = True,
        episodic_life: bool = False,
        life_loss_info: bool = False,
        autoreset_mode: AutoresetMode | str = AutoresetMode.SAME_STEP,
    ):
        if not _power_on_state(state):
            raise ValueError(
                "AtariVecEnv does not support Stable Retro save states; "
                "use state=State.NONE or use RetroVecEnv for the Stella state contract",
            )
        if obs_resize is None:
            obs_resize = (84, 84)
        try:
            obs_height, obs_width = obs_resize
        except (TypeError, ValueError) as exc:
            raise ValueError("obs_resize must be a (height, width) pair") from exc
        obs_height = int(obs_height)
        obs_width = int(obs_width)
        if obs_height <= 0 or obs_width <= 0:
            raise ValueError("obs_resize dimensions must be positive")
        if use_restricted_actions is Actions.ALL:
            full_action_space = True
        elif use_restricted_actions in (Actions.FILTERED, Actions.DISCRETE):
            full_action_space = False
        else:
            raise ValueError(
                "AtariVecEnv supports Actions.ALL, Actions.FILTERED, or Actions.DISCRETE",
            )
        if not isinstance(reward_clip, bool):
            raise TypeError("AtariVecEnv reward_clip must be a bool")
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")

        self.stable_retro_game = str(game)
        self.ale_game = ale_game_id(game)
        self.autoreset_mode = _normalize_autoreset_mode(autoreset_mode)
        self.metadata = dict(type(self).metadata)
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.num_envs = int(num_envs)
        self.batch_size = self.num_envs if int(batch_size) == 0 else int(batch_size)
        if not 0 < self.batch_size <= self.num_envs:
            raise ValueError("batch_size must be zero or between 1 and num_envs")
        self._closed = False

        rom_path = str(roms.get_rom_path(self.ale_game))
        self.ale = _retro._AtariVecEnv(
            rom_path=rom_path,
            num_envs=self.num_envs,
            batch_size=int(batch_size),
            num_threads=int(num_threads),
            thread_affinity_offset=int(thread_affinity_offset),
            max_episode_steps=int(max_episode_steps),
            repeat_action_probability=float(sticky_action_prob),
            full_action_space=full_action_space,
            autoreset_mode=self.autoreset_mode.value,
            img_height=obs_height,
            img_width=obs_width,
            grayscale=bool(obs_grayscale),
            stack_num=int(frame_stack),
            frameskip=int(frame_skip),
            maxpool=bool(maxpool_last_two),
            noop_max=int(noop_reset_max),
            episodic_life=bool(episodic_life),
            life_loss_info=bool(life_loss_info),
            reward_clipping=reward_clip,
            use_fire_reset=bool(use_fire_reset),
        )

        obs_shape = (int(frame_stack), obs_height, obs_width)
        if not obs_grayscale:
            obs_shape += (3,)
        self.single_observation_space = Box(
            shape=obs_shape,
            low=0,
            high=255,
            dtype=np.uint8,
        )
        self.single_action_space = Discrete(len(self.ale.get_action_set()))
        self.observation_space = batch_space(
            self.single_observation_space,
            self.batch_size,
        )
        self.action_space = batch_space(self.single_action_space, self.batch_size)

    def _reset_indices(self, options: dict[str, Any] | None) -> np.ndarray:
        if options is None or "reset_mask" not in options:
            return np.arange(self.num_envs, dtype=np.int32)
        reset_mask = options["reset_mask"]
        if not isinstance(reset_mask, np.ndarray):
            raise TypeError("reset_mask must be a numpy.ndarray")
        if reset_mask.dtype != np.bool_:
            raise TypeError("reset_mask must have dtype numpy.bool_")
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(f"reset_mask must have shape ({self.num_envs},)")
        return np.flatnonzero(reset_mask).astype(np.int32, copy=False)

    def _reset_seeds(
        self,
        seed: int | Sequence[int | None] | np.ndarray | None,
        indices: np.ndarray,
    ) -> np.ndarray:
        if seed is None:
            return np.full(indices.size, -1, dtype=np.int32)
        if isinstance(seed, Integral):
            values = [int(seed) + int(index) for index in indices]
        else:
            if isinstance(seed, np.ndarray) and seed.ndim != 1:
                raise ValueError("seed array must be one-dimensional")
            try:
                supplied = list(seed)
            except TypeError as exc:
                raise TypeError(
                    "seed must be None, an integer, or a one-dimensional seed sequence",
                ) from exc
            if len(supplied) == self.num_envs:
                values = [supplied[int(index)] for index in indices]
            elif len(supplied) == indices.size:
                values = supplied
            else:
                raise ValueError(
                    "seed sequence must have length num_envs or the number of reset lanes",
                )
            values = [-1 if value is None else int(value) for value in values]
        seeds = np.asarray(values, dtype=np.int64)
        if np.any(seeds < -1) or np.any(seeds > np.iinfo(np.uint32).max):
            raise ValueError("seed values must be None or integers in [0, 2**32 - 1]")
        # ALE stores random_seed as a non-negative signed int. Accept the full
        # uint32 domain used by NumPy SeedSequence and fold only its high bit.
        # The -1 sentinel remains reserved for an unseeded reset.
        seeded = seeds >= 0
        seeds[seeded] &= np.iinfo(np.int32).max
        return seeds.astype(np.int32, copy=False)

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ):
        indices = self._reset_indices(options)
        seeds = self._reset_seeds(seed, indices)
        return self.ale.reset(indices.tolist(), seeds.tolist())

    def step(self, actions: np.ndarray):
        return self.ale.step(np.asarray(actions, dtype=np.int64))

    def send(self, actions: np.ndarray):
        self.ale.send(np.asarray(actions, dtype=np.int64))

    def recv(self):
        return self.ale.recv()

    def close(self, **kwargs):
        if not self._closed:
            self.ale = None
            self._closed = True


__all__ = ["AtariVecEnv", "ale_game_id"]

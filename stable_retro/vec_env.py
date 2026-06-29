"""Native vector environment for stable-retro rollouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import stable_retro.data as retro_data
from stable_retro.enums import Actions, Observations, State

try:
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover - import remains cheap without SB3.
    VecEnv = object


class RetroVecEnv(VecEnv):
    """SB3-compatible native vector env for stable-retro rollouts.

    This is the supported high-throughput path. C++ owns the emulator pool,
    frame skip, preprocessing, frame stacking, autoreset, reward/done
    evaluation, and batched observation buffer.

    ``state`` accepts a single state name, a sequence of one state per env slot,
    or a mapping of state names to positive sampling weights. Mapping weights are
    normalized and sampled independently for each env on every episode reset.

    Native info-transition termination is opt-in and game/config-specific.
    Pass done_on_info={"name": ["key", "decrease"]} to terminate and autoreset
    only lanes whose post-reset baseline changes as requested. Supported ops
    are "change", "increase", and "decrease"; keys may be a string or a
    sequence of strings. The legacy terminate_on_life_loss=True,
    life_variable="lives" option is compiled into a "life_loss" decrease rule.
    If both forms provide "life_loss", the explicit done_on_info rule wins.
    Fired rules are reported in info["done_on_info"] with list-shaped "keys",
    "prev", and "next" values, including one-element lists for single-key rules.

    With copy_observations=False, returned observations are double-buffered so
    the previous observation survives the next step for SB3 rollout collection.
    unsafe_zero_copy=True restores single-buffer aliasing for benchmarks only.
    """

    def __init__(
        self,
        game,
        state=State.DEFAULT,
        scenario=None,
        info=None,
        use_restricted_actions=Actions.FILTERED,
        record=False,
        players=1,
        inttype=retro_data.Integrations.STABLE,
        obs_type=Observations.IMAGE,
        render_mode="human",
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
        copy_observations: bool = True,
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
    ):
        if VecEnv is object:
            raise ImportError(
                "RetroVecEnv requires stable-baselines3 to be installed",
            )
        import stable_retro as retro
        from stable_retro import _retro

        (
            state_values,
            state_labels,
            state_probs,
            state_collection,
        ) = self._resolve_state_config(
            retro,
            game,
            num_envs,
            state,
        )
        self.waiting = False
        self.closed = False
        self._actions = None
        self._observations = None

        env_kwargs = {
            "use_restricted_actions": use_restricted_actions,
            "record": record,
            "players": players,
            "obs_type": obs_type,
            "render_mode": render_mode,
            "obs_resize": obs_resize,
            "obs_crop": obs_crop,
            "obs_grayscale": obs_grayscale,
            "obs_resize_algorithm": obs_resize_algorithm,
            "frame_skip": frame_skip,
            "frame_stack": frame_stack,
            "maxpool_last_two": maxpool_last_two,
            "noop_reset_max": noop_reset_max,
            "sticky_action_prob": sticky_action_prob,
            "reward_clip": reward_clip,
        }
        info_mode = str(info_mode)
        if isinstance(info_keys, str):
            raise ValueError("info_keys must be a sequence of strings, not a string")
        if info_keys is not None:
            info_keys = [str(key) for key in info_keys]
        unsafe_zero_copy = bool(unsafe_zero_copy)
        terminate_on_life_loss = bool(terminate_on_life_loss)
        if terminate_on_life_loss:
            if life_variable is None or str(life_variable) == "":
                raise ValueError(
                    "life_variable is required when terminate_on_life_loss=True",
                )
            life_variable = str(life_variable)
        elif life_variable is None:
            life_variable = ""
        else:
            life_variable = str(life_variable)
        done_on_info_rules = self._normalize_done_on_info(
            done_on_info,
            terminate_on_life_loss=terminate_on_life_loss,
            life_variable=life_variable,
        )
        obs_layout = str(obs_layout).lower()
        if obs_layout not in {"hwc", "chw"}:
            raise ValueError("obs_layout must be 'hwc' or 'chw'")
        self.obs_layout = obs_layout
        self.copy_observations = bool(copy_observations)
        self.unsafe_zero_copy = unsafe_zero_copy
        if self.copy_observations and self.unsafe_zero_copy:
            raise ValueError(
                "unsafe_zero_copy=True is only valid with copy_observations=False",
            )
        self.render_mode = render_mode
        if players != 1:
            raise ValueError("RetroVecEnv currently supports players=1")
        if record:
            raise ValueError("RetroVecEnv does not support movie recording")
        if obs_type != retro.Observations.IMAGE:
            raise ValueError(
                "RetroVecEnv currently supports image observations only",
            )

        info_path = self._resolve_info_path(retro, game, info, inttype)
        scenario_path = self._resolve_scenario_path(retro, game, scenario, inttype)
        template = self._make_template_env(
            retro,
            game,
            state_values[0],
            inttype,
            rom_path,
            info_path,
            scenario_path,
            env_kwargs,
        )
        try:
            if template._rotation_steps() != 0:
                raise ValueError(
                    "RetroVecEnv does not support rotated screens",
                )
            width, height = template.em.get_resolution()
            crop = template._effective_crop(0, height, width)
            initial_state = template.initial_state if template.initial_state else None
            self.action_space = template.action_space
            self.observation_space = self._observation_space_for_layout(
                template.observation_space,
                obs_layout,
            )
            self.num_buttons = template.num_buttons
            self.button_combos = [
                [int(action) for action in combo] for combo in template.button_combos
            ]
            self.use_restricted_actions = template.use_restricted_actions
            self._filter_actions = self.use_restricted_actions == retro.Actions.FILTERED
            reward_clip, reward_low, reward_high = self._reward_clip_config(template)
        finally:
            template.close()

        initial_state_labels = None
        initial_state_weights = None
        if state_collection:
            initial_states = [initial_state]
            state_cache = {state_labels[0]: initial_state}
            for value, label in zip(state_values[1:], state_labels[1:]):
                cached = state_cache.get(label)
                if cached is not None:
                    initial_states.append(cached)
                    continue
                state_template = self._make_template_env(
                    retro,
                    game,
                    value,
                    inttype,
                    rom_path,
                    info_path,
                    scenario_path,
                    env_kwargs,
                )
                try:
                    serialized = (
                        state_template.initial_state
                        if state_template.initial_state
                        else None
                    )
                finally:
                    state_template.close()
                if not serialized:
                    raise ValueError(
                        f"state {label!r} did not resolve to a non-empty state",
                    )
                state_cache[label] = serialized
                initial_states.append(serialized)
            if any(not value for value in initial_states):
                raise ValueError("states must resolve to non-empty start states")
            initial_state = initial_states
            initial_state_labels = state_labels
            initial_state_weights = state_probs
        elif initial_state is not None:
            initial_state_labels = state_labels

        resolved_rom_path = rom_path or retro.data.get_original_romfile_path(
            game,
            inttype,
        )
        if num_threads is None:
            num_threads = num_envs
        self.native = _retro.NativeVectorEnv(
            int(num_envs),
            str(resolved_rom_path),
            str(info_path),
            str(scenario_path),
            initial_state,
            int(self.num_buttons),
            int(env_kwargs.get("frame_skip", 1)),
            int(env_kwargs.get("frame_stack", 1)),
            crop,
            env_kwargs.get("obs_resize", None),
            bool(env_kwargs.get("obs_grayscale", False)),
            str(env_kwargs.get("obs_resize_algorithm", "nearest")),
            bool(env_kwargs.get("maxpool_last_two", False)),
            int(env_kwargs.get("noop_reset_max", 0)),
            float(env_kwargs.get("sticky_action_prob", 0.0)),
            bool(self._filter_actions),
            bool(reward_clip),
            float(reward_low),
            float(reward_high),
            int(num_threads),
            info_mode,
            unsafe_zero_copy,
            obs_layout,
            info_keys,
            terminate_on_life_loss,
            life_variable,
            done_on_info_rules,
            initial_state_labels,
            initial_state_weights,
        )
        super().__init__(int(num_envs), self.observation_space, self.action_space)
        self.initial_state_names = tuple(self.native.initial_state_names)
        self._active_state_indices = self.native.active_state_indices()
        self._active_state_indices.setflags(write=False)

    def active_state_indices(self):
        """Return a read-only int32 NumPy view of active initial-state indices.

        The returned array is owned by the native vector env and mutates in place
        after ``reset()`` and after per-lane automatic resets inside
        ``step_wait()``. Copy it when a stable snapshot is needed.
        Lanes without a serialized initial state report ``-1``.
        """
        return self._active_state_indices

    def active_states(self):
        """Return active initial-state names for each lane."""
        names = self.initial_state_names
        return tuple(
            None if int(index) < 0 else names[int(index)]
            for index in self._active_state_indices
        )

    @staticmethod
    def _normalize_done_on_info(
        done_on_info,
        *,
        terminate_on_life_loss: bool,
        life_variable: str,
    ):
        rules = {}
        if terminate_on_life_loss:
            rules["life_loss"] = ((life_variable,), "decrease")
        if done_on_info is not None:
            if not isinstance(done_on_info, Mapping):
                raise ValueError("done_on_info must be a mapping of rule names to (keys, op)")
            for raw_name, spec in done_on_info.items():
                name = str(raw_name)
                if not name:
                    raise ValueError("done_on_info rule names must not be empty")
                if not (
                    isinstance(spec, Sequence)
                    and not isinstance(spec, (str, bytes, bytearray))
                    and len(spec) == 2
                ):
                    raise ValueError(
                        "done_on_info values must be (key_or_keys, op) pairs",
                    )
                raw_keys, raw_op = spec
                op = str(raw_op)
                if op not in {"change", "increase", "decrease"}:
                    raise ValueError(
                        "done_on_info ops must be 'change', 'increase', or 'decrease'",
                    )
                if isinstance(raw_keys, str):
                    keys = (raw_keys,)
                elif (
                    isinstance(raw_keys, Sequence)
                    and not isinstance(raw_keys, (bytes, bytearray))
                ):
                    keys = tuple(str(key) for key in raw_keys)
                else:
                    raise ValueError(
                        "done_on_info keys must be a string or sequence of strings",
                    )
                if not keys or any(not key for key in keys):
                    raise ValueError("done_on_info rules must reference at least one key")
                rules[name] = (keys, op)
        if not rules:
            return None
        return tuple((name, keys, op) for name, (keys, op) in rules.items())

    @staticmethod
    def _observation_space_for_layout(observation_space, obs_layout):
        if obs_layout == "hwc":
            return observation_space
        if len(observation_space.shape) != 3:
            raise ValueError("obs_layout='chw' requires image observations")
        height, width, channels = observation_space.shape
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, height, width),
            dtype=np.uint8,
        )

    @staticmethod
    def _resolve_state_config(retro, game, num_envs, state):
        if isinstance(state, Mapping):
            states = list(state.keys())
            state_probs = list(state.values())
            state_collection = True
        elif (
            isinstance(state, Sequence)
            and not isinstance(state, (str, bytes, bytearray))
        ):
            states = list(state)
            state_probs = None
            state_collection = True
        else:
            return [state], [str(state)], None, False

        if not states:
            raise ValueError("state must contain at least one state")

        available_states = set(retro.data.list_states(game))
        state_values = []
        state_labels = []
        for value in states:
            label = str(value).strip()
            if not label:
                raise ValueError("state must not contain empty state names")
            if (
                value not in (retro.State.DEFAULT, retro.State.NONE)
                and label not in available_states
                and not Path(label).exists()
            ):
                raise ValueError(f"unknown state {label!r} for game {game}")
            state_values.append(value)
            state_labels.append(label)

        if state_probs is None:
            if len(state_values) != int(num_envs):
                raise ValueError(
                    "state sequence length must match num_envs",
                )
            return state_values, state_labels, None, state_collection

        probs = list(state_probs)
        if len(probs) != len(state_values):
            raise ValueError("state weight count must match state count")

        normalized_probs = []
        for prob in probs:
            value = float(prob)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("state weights must contain positive finite numbers")
            normalized_probs.append(value)
        total = math.fsum(normalized_probs)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("state weights must sum to a positive finite number")
        normalized_probs = [value / total for value in normalized_probs]
        return state_values, state_labels, normalized_probs, state_collection

    @staticmethod
    def _resolve_info_path(retro, game, info, inttype):
        if info is None:
            info = "data"
        if str(info).endswith(".json"):
            return str(info)
        return retro.data.get_file_path(game, str(info) + ".json", inttype)

    @staticmethod
    def _resolve_scenario_path(retro, game, scenario, inttype):
        if scenario is None:
            scenario = "scenario"
        if str(scenario).endswith(".json"):
            return str(scenario)
        return retro.data.get_file_path(game, str(scenario) + ".json", inttype)

    @staticmethod
    def _reward_clip_config(template):
        reward_clip = template._reward_clip
        if not reward_clip:
            return False, -1.0, 1.0
        if reward_clip is True:
            return True, -1.0, 1.0
        low, high = reward_clip
        return True, float(low), float(high)

    @staticmethod
    def _make_template_env(
        retro,
        game,
        state,
        inttype,
        rom_path,
        info_path,
        scenario_path,
        env_kwargs,
    ):
        if rom_path is None:
            return retro.make(
                game,
                state=state,
                inttype=inttype,
                info=info_path,
                scenario=scenario_path,
                **env_kwargs,
            )
        original_get_romfile_path = retro.data.get_romfile_path
        original_get_original_romfile_path = retro.data.get_original_romfile_path
        original_get_file_path = retro.data.get_file_path
        try:
            retro.data.get_romfile_path = lambda *_args, **_kwargs: str(rom_path)
            retro.data.get_original_romfile_path = lambda *_args, **_kwargs: str(
                rom_path,
            )
            retro.data.get_file_path = lambda _game, file, *_args, **_kwargs: {
                "data.json": str(info_path),
                "scenario.json": str(scenario_path),
            }.get(file, original_get_file_path(_game, file, *_args, **_kwargs))
            return retro.make(
                game,
                state=state,
                inttype=inttype,
                info=info_path,
                scenario=scenario_path,
                **env_kwargs,
            )
        finally:
            retro.data.get_romfile_path = original_get_romfile_path
            retro.data.get_original_romfile_path = original_get_original_romfile_path
            retro.data.get_file_path = original_get_file_path

    def _obs(self):
        if self.copy_observations:
            return self._observations.copy()
        return self._observations

    def _actions_to_masks(self, actions):
        import stable_retro as retro

        if self.use_restricted_actions in (retro.Actions.ALL, retro.Actions.FILTERED):
            masks = np.asarray(actions, dtype=np.uint8)
            return masks.reshape((self.num_envs, self.num_buttons))

        masks = np.zeros((self.num_envs, self.num_buttons), dtype=np.uint8)
        if self.use_restricted_actions == retro.Actions.DISCRETE:
            for env_idx, action in enumerate(np.asarray(actions).reshape(-1)):
                value = int(action)
                action_bits = 0
                for combo in self.button_combos:
                    current = value % len(combo)
                    value //= len(combo)
                    action_bits |= combo[current]
                for key in range(self.num_buttons):
                    masks[env_idx, key] = (action_bits >> key) & 1
            return masks

        for env_idx, action in enumerate(np.asarray(actions)):
            action_bits = 0
            for key, value in enumerate(action[: len(self.button_combos)]):
                combo = self.button_combos[key]
                action_bits |= combo[int(value)]
            for key in range(self.num_buttons):
                masks[env_idx, key] = (action_bits >> key) & 1
        return masks

    def reset(self):
        seeds = None
        if self._seeds:
            seeds = [
                None if value is None else int(value)
                for value in self._seeds
            ]
            if all(value is None for value in seeds):
                seeds = None
        obs, infos = self.native.reset(seeds)
        self._observations = np.asarray(obs, dtype=np.uint8)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        return self._obs()

    def step_async(self, actions):
        self._actions = self._actions_to_masks(actions)
        self.waiting = True

    def step_wait(self):
        obs, rewards, dones, infos = self.native.step(self._actions)
        self._actions = None
        self.waiting = False
        self._observations = np.asarray(obs, dtype=np.uint8)
        return (
            self._obs(),
            np.array(rewards, dtype=np.float32, copy=True),
            np.array(dones, dtype=bool, copy=True),
            list(infos),
        )

    def close(self):
        self.closed = True

    def get_images(self):
        return []

    def render(self, mode: str | None = None):
        return None

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        method = getattr(self, method_name)
        return [
            method(*method_args, **method_kwargs) for _ in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]


__all__ = ["RetroVecEnv"]

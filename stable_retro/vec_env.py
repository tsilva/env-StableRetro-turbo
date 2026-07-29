"""Gymnasium vector environments for stable-retro rollouts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import stable_retro.data as retro_data
from stable_retro.enums import Actions, Observations
from stable_retro.action_tables import ActionTable

_SERIALIZED_UNSET = object()


class _DefaultState:
    def __repr__(self):
        return "State.DEFAULT"


_STATE_UNSET = _DefaultState()


def _normalize_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _normalize_positive_int(value, name):
    value = _normalize_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_nonnegative_int(value, name):
    value = _normalize_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _normalize_probability(value, name):
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0.0 and 1.0") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return probability


def _normalize_obs_resize(value):
    if value is None:
        return None
    try:
        height, width = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("obs_resize must be a (height, width) pair") from exc
    return (
        _normalize_positive_int(height, "obs_resize height"),
        _normalize_positive_int(width, "obs_resize width"),
    )


def _normalize_obs_crop(value):
    if value is None:
        return None
    try:
        top, bottom, left, right = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "obs_crop must be a (top, bottom, left, right) tuple",
        ) from exc
    return tuple(
        _normalize_nonnegative_int(item, name)
        for item, name in zip(
            (top, bottom, left, right),
            ("obs_crop top", "obs_crop bottom", "obs_crop left", "obs_crop right"),
            strict=True,
        )
    )


def _normalize_obs_crop_fill(value):
    fill = _normalize_int(value, "obs_crop_fill")
    if not 0 <= fill <= 255:
        raise ValueError("obs_crop_fill must be between 0 and 255")
    return fill


def _normalize_obs_resize_algorithm(value):
    algorithm = str(value).lower()
    if algorithm not in {"nearest", "bilinear", "area"}:
        raise ValueError(
            "obs_resize_algorithm must be one of: nearest, bilinear, area",
        )
    return algorithm


def _normalize_reward_clip(value):
    if value is False or value is True:
        return value
    try:
        low, high = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward_clip must be a bool or (low, high) pair") from exc
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward_clip bounds must be finite numbers") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(
            "reward_clip bounds must be finite numbers with low <= high",
        )
    return low, high


class RetroVecEnv(VectorEnv):
    """Gymnasium vector env for stable-retro rollouts.

    This is the supported high-throughput path. C++ owns the emulator pool,
    frame skip, preprocessing, frame stacking, reward/done evaluation, and
    batched observation buffer.

    Terminated lanes keep their terminal observation and cannot be stepped
    again until selected with
    ``reset(options={"reset_mask": mask})``. Masked reset leaves every
    unselected lane untouched.

    ``state`` accepts one saved-state name. ``state_catalog`` accepts an ordered
    collection of every saved state that may be selected for this env instance.
    Selected lanes load catalog entries through ``options["state_indices"]``.

    The disabled ``use_fire_reset`` default preserves scalar ``RetroEnv`` reset
    behavior. On Stella-backed Atari environments, ``use_fire_reset=True``
    presses FIRE for one native frame after each full-episode reset when FIRE
    is available, then releases it before reset no-ops.

    obs_copy="safe_view" returns double-buffered observation views so the
    previous observation survives the next step for rollout collection.
    obs_copy="unsafe_view" restores single-buffer aliasing for benchmarks only.
    """

    metadata = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["rgb_array"],
        "turbo_api_version": 1,
    }

    def __init__(
        self,
        game,
        state=_STATE_UNSET,
        scenario=None,
        info=None,
        use_restricted_actions: Actions | str | ActionTable = Actions.FILTERED,
        record=False,
        players=1,
        inttype=retro_data.Integrations.STABLE,
        obs_type=Observations.IMAGE,
        render_mode="human",
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
        obs_copy="copy",
        obs_resize=None,
        obs_crop=None,
        obs_crop_mode: Literal["remove", "mask"] = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale=False,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        state_catalog=None,
    ):
        import stable_retro as retro
        from stable_retro import _retro

        num_envs = _normalize_positive_int(num_envs, "num_envs")
        if num_threads is not None:
            num_threads = _normalize_positive_int(num_threads, "num_threads")
        obs_resize = _normalize_obs_resize(obs_resize)
        obs_crop = _normalize_obs_crop(obs_crop)
        obs_crop_fill = _normalize_obs_crop_fill(obs_crop_fill)
        obs_resize_algorithm = _normalize_obs_resize_algorithm(obs_resize_algorithm)
        frame_skip = _normalize_positive_int(frame_skip, "frame_skip")
        frame_stack = _normalize_positive_int(frame_stack, "frame_stack")
        noop_reset_max = _normalize_nonnegative_int(
            noop_reset_max,
            "noop_reset_max",
        )
        use_fire_reset = bool(use_fire_reset)
        sticky_action_prob = _normalize_probability(
            sticky_action_prob,
            "sticky_action_prob",
        )
        reward_clip = _normalize_reward_clip(reward_clip)

        self.closed = False
        self.autoreset_mode = AutoresetMode.DISABLED
        self._observations = None
        self._seeds = [None for _ in range(num_envs)]
        self._options = [None for _ in range(num_envs)]
        self.use_fire_reset = use_fire_reset

        info_filter_mode, info_filter_keys = self._normalize_info_filter(info_filter)
        self._info_filter_mode = info_filter_mode
        copy_obs, unsafe_view = self._normalize_obs_copy(obs_copy)
        obs_crop_mode = self._normalize_obs_crop_mode(obs_crop_mode)

        env_kwargs = {
            "use_restricted_actions": use_restricted_actions,
            "record": record,
            "players": players,
            "obs_type": obs_type,
            "render_mode": render_mode,
        }
        info_path = self._resolve_info_path(retro, game, info, inttype)
        scenario_path = self._resolve_scenario_path(retro, game, scenario, inttype)
        state_values, state_labels, explicit_catalog = self._resolve_state_config(
            retro,
            game,
            state,
            state_catalog,
        )
        obs_layout = str(obs_layout).lower()
        if obs_layout not in {"hwc", "chw"}:
            raise ValueError("obs_layout must be 'hwc' or 'chw'")
        self.obs_layout = obs_layout
        self.obs_copy = (
            "unsafe_view"
            if unsafe_view
            else "copy" if copy_obs else "safe_view"
        )
        self.observation_ownership = (
            "owned"
            if self.obs_copy == "copy"
            else "unsafe_view" if self.obs_copy == "unsafe_view" else "safe_view"
        )
        self.observation_buffer_depth = (
            None
            if self.obs_copy == "copy"
            else 1 if self.obs_copy == "unsafe_view" else 2
        )
        self._copy_obs = copy_obs
        self._unsafe_view = unsafe_view
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.obs_resize_algorithm = obs_resize_algorithm
        self.obs_grayscale = bool(obs_grayscale)
        self.maxpool_last_two = bool(maxpool_last_two)
        self.noop_reset_max = noop_reset_max
        self.sticky_action_prob = sticky_action_prob
        self.render_mode = render_mode
        self.viewer = None
        if players != 1:
            raise ValueError("RetroVecEnv currently supports players=1")
        if record:
            raise ValueError("RetroVecEnv does not support movie recording")
        if obs_type != retro.Observations.IMAGE:
            raise ValueError(
                "RetroVecEnv currently supports image observations only",
            )

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
            integration_crop = tuple(int(value) for value in template.data.crop_info(0))
            crop = self._effective_crop(
                template,
                height,
                width,
                obs_crop,
                obs_crop_mode,
            )
            remove_crop = obs_crop if obs_crop_mode == "remove" else None
            crop_mask = obs_crop if obs_crop_mode == "mask" else None
            source_height = height - crop[0] - crop[1]
            source_width = width - crop[2] - crop[3]
            initial_state = self._initial_state(template)
            self.single_action_space = template.action_space
            self.system = template.system
            self.buttons = tuple(template.buttons)
            self.single_observation_space = self._observation_space_for_layout(
                self._vector_observation_space(
                    height,
                    width,
                    crop,
                    obs_resize,
                    bool(obs_grayscale),
                    frame_stack,
                ),
                obs_layout,
            )
            self.action_space = batch_space(self.single_action_space, int(num_envs))
            self.observation_space = batch_space(
                self.single_observation_space,
                num_envs,
            )
            self.num_buttons = template.num_buttons
            fire_button = self._fire_reset_button(template, use_fire_reset)
            self.button_combos = [
                [int(action) for action in combo] for combo in template.button_combos
            ]
            self.action_mode = template.action_mode
            self.action_preset = template.action_preset
            self.action_table = template.action_table
            self.action_meanings = template.action_meanings
            self.action_table_hash = template.action_table_hash
            self._custom_action_masks = template._custom_action_masks
            self.use_restricted_actions = template.use_restricted_actions
            self._filter_actions = self.action_mode == "filtered"
            reward_clip, reward_low, reward_high = self._reward_clip_config(reward_clip)
            template_signal_names = tuple(
                str(name) for name in template.data.list_variables()
            )
        finally:
            template.close()

        initial_state, initial_state_labels = self._resolve_initial_state_payload(
            retro,
            game,
            inttype,
            rom_path,
            info_path,
            scenario_path,
            env_kwargs,
            first_initial_state=initial_state,
            resolved_config=(state_values, state_labels, explicit_catalog),
        )
        if initial_state is None:
            state_payloads = ()
        elif isinstance(initial_state, Sequence) and not isinstance(
            initial_state, (str, bytes, bytearray)
        ):
            state_payloads = tuple(bytes(payload) for payload in initial_state)
        else:
            state_payloads = (bytes(initial_state),)
        labels = tuple(initial_state_labels or ())
        if len(labels) != len(state_payloads):
            labels = tuple(f"state-{index}" for index in range(len(state_payloads)))
        self.initial_state_assets = tuple(
            {"name": str(label), "sha256": hashlib.sha256(payload).hexdigest()}
            for label, payload in zip(labels, state_payloads)
        )

        resolved_rom_path = rom_path or retro.data.get_original_romfile_path(
            game,
            inttype,
        )
        self.rom_path = str(resolved_rom_path)
        self.info_path = str(info_path)
        self.scenario_path = str(scenario_path)
        if num_threads is None:
            num_threads = num_envs
        self.num_threads = int(num_threads)
        self.reward_clip = bool(reward_clip)
        self.native = _retro._RetroVecEnv(
            num_envs,
            str(resolved_rom_path),
            str(info_path),
            str(scenario_path),
            initial_state,
            int(self.num_buttons),
            frame_skip,
            frame_stack,
            integration_crop,
            remove_crop,
            source_height,
            source_width,
            obs_resize,
            bool(obs_grayscale),
            obs_resize_algorithm,
            bool(maxpool_last_two),
            noop_reset_max,
            fire_button,
            sticky_action_prob,
            bool(self._filter_actions),
            bool(reward_clip),
            float(reward_low),
            float(reward_high),
            num_threads,
            info_filter_mode,
            unsafe_view,
            obs_layout,
            info_filter_keys,
            initial_state_labels,
            crop_mask,
            obs_crop_fill,
        )
        native_shape = tuple(int(value) for value in self.native.observation_shape())
        if native_shape != self.single_observation_space.shape:
            raise RuntimeError(
                "native observation shape does not match the declared "
                f"single observation space: {native_shape} != "
                f"{self.single_observation_space.shape}",
            )
        self.num_envs = num_envs
        self._state_catalog = tuple(self.native.state_catalog)
        self._active_state_indices = self.native.active_state_indices()
        self._active_state_indices.setflags(write=False)
        self.supports_live_snapshots = bool(self.native.supports_live_snapshots)
        self.live_snapshots_deterministic = self.supports_live_snapshots
        selected_signal_names = (
            template_signal_names
            if info_filter_keys is None
            else tuple(
                name for name in info_filter_keys if name in template_signal_names
            )
        )
        self.signal_schema = MappingProxyType(
            {
                name: MappingProxyType(
                    {
                        "dtype": np.dtype(np.int64),
                        "shape": (),
                        "available_on_reset": info_filter_mode == "all",
                        "available_on_step": info_filter_mode != "none",
                    }
                )
                for name in selected_signal_names
            }
            if info_filter_mode != "none"
            else {}
        )
        self.capabilities = MappingProxyType(
            {
                "supported_action_modes": (
                    "all",
                    "filtered",
                    "discrete",
                    "multi_discrete",
                    "custom_discrete",
                ),
                "supported_observation_layouts": ("chw", "hwc"),
                "supported_resize_algorithms": ("nearest", "bilinear", "area"),
                "supported_observation_copy_modes": (
                    "copy",
                    "safe_view",
                    "unsafe_view",
                ),
                "supports_maxpool_last_two": True,
                "supports_sticky_action_prob": True,
                "supports_reward_clipping": True,
                "supports_noop_reset": True,
                "supports_state_catalog": True,
                "supports_live_snapshots": self.supports_live_snapshots,
                "supports_per_lane_rgb": True,
            }
        )
        self._initialized = np.zeros(self.num_envs, dtype=np.bool_)

    @staticmethod
    def _fire_reset_button(template, enabled):
        """Return the Stella FIRE button index, or -1 when unavailable."""
        if not enabled or template.system != "Atari2600":
            return -1
        try:
            fire_button = template.buttons.index("BUTTON")
        except ValueError:
            return -1
        if template.action_mode == "all":
            return fire_button
        if template.action_mode == "custom_discrete":
            fire_mask = 1 << fire_button
            if any(action[0] & fire_mask for action in template._custom_action_masks):
                return fire_button
            return -1
        fire_mask = 1 << fire_button
        if any(
            int(action) & fire_mask
            for combo in template.button_combos
            for action in combo
        ):
            return fire_button
        return -1

    @property
    def state_catalog(self) -> tuple[str, ...]:
        """Return the immutable ordered catalog of selectable saved states."""
        return self._state_catalog

    def active_state_indices(self) -> np.ndarray:
        """Return a read-only int32 NumPy view of active initial-state indices.

        The returned array is owned by this env and mutates in place
        after ``reset()``. Copy it when a stable snapshot is needed.
        Lanes without a serialized initial state report ``-1``.
        """
        return self._active_state_indices

    @staticmethod
    def _normalize_obs_copy(obs_copy):
        if isinstance(obs_copy, bool):
            raise ValueError(
                "obs_copy must be 'copy', 'safe_view', or 'unsafe_view'",
            )
        mode = str(obs_copy).lower()
        if mode == "copy":
            return True, False
        if mode == "safe_view":
            return False, False
        if mode == "unsafe_view":
            return False, True
        raise ValueError("obs_copy must be 'copy', 'safe_view', or 'unsafe_view'")

    @staticmethod
    def _normalize_obs_crop_mode(obs_crop_mode):
        mode = str(obs_crop_mode).lower()
        if mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        return mode

    @staticmethod
    def _normalize_obs_crop_fill(obs_crop_fill):
        fill = int(obs_crop_fill)
        if fill < 0 or fill > 255:
            raise ValueError("obs_crop_fill must be between 0 and 255")
        return fill

    @classmethod
    def _normalize_info_filter(cls, info_filter):
        if info_filter is None:
            return "all", None
        if isinstance(info_filter, str):
            mode = str(info_filter)
            if mode not in {"terminal", "all", "none"}:
                raise ValueError("info_filter mode must be terminal, all, or none")
            return mode, None
        if not isinstance(info_filter, Mapping):
            raise ValueError(
                "info_filter must be a mode string or a mapping with mode/keys",
            )
        unknown = set(info_filter) - {"mode", "keys"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"unknown info_filter keys: {names}")
        mode = str(info_filter.get("mode", "all"))
        if mode not in {"terminal", "all", "none"}:
            raise ValueError("info_filter mode must be terminal, all, or none")
        keys = cls._normalize_info_filter_keys(info_filter.get("keys", None))
        return mode, keys

    @staticmethod
    def _normalize_info_filter_keys(info_filter_keys):
        if isinstance(info_filter_keys, str):
            raise ValueError(
                "info_filter keys must be a sequence of strings, not a string",
            )
        if info_filter_keys is None:
            return None
        return [str(key) for key in info_filter_keys]

    @staticmethod
    def _effective_crop(template, raw_height, raw_width, obs_crop, obs_crop_mode):
        x, y, width, height = template.data.crop_info(0)
        right_edge = raw_width if not width or x + width > raw_width else x + width
        bottom_edge = (
            raw_height if not height or y + height > raw_height else y + height
        )
        top, left = int(y), int(x)
        if obs_crop is not None and obs_crop_mode == "remove":
            obs_top, obs_bottom, obs_left, obs_right = obs_crop
            top += obs_top
            left += obs_left
            bottom_edge -= obs_bottom
            right_edge -= obs_right
        if top >= bottom_edge or left >= right_edge:
            raise ValueError("obs_crop removes the entire observation")
        return (
            top,
            int(raw_height - bottom_edge),
            left,
            int(raw_width - right_edge),
        )

    @staticmethod
    def _vector_observation_space(
        raw_height,
        raw_width,
        crop,
        obs_resize,
        obs_grayscale,
        frame_stack,
    ):
        top, bottom, left, right = crop
        height = raw_height - top - bottom
        width = raw_width - left - right
        if obs_resize is not None:
            height, width = obs_resize
        channels = (1 if obs_grayscale else 3) * frame_stack
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels),
            dtype=np.uint8,
        )

    @staticmethod
    def _initial_state(template):
        initial_state = template.initial_state if template.initial_state else None
        if initial_state is not None and template.system == "Atari2600":
            from stable_retro.stella_state import migrate_legacy_state

            initial_state = migrate_legacy_state(template.em, initial_state)
        return initial_state

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
    def _resolve_state_config(retro, game, state, state_catalog):
        if state_catalog is None:
            if state is _STATE_UNSET:
                state = retro.State.DEFAULT
            if isinstance(state, Mapping) or (
                isinstance(state, Sequence)
                and not isinstance(state, (str, bytes, bytearray))
            ):
                raise TypeError(
                    "state must be a single state; use state_catalog for multiple states",
                )
            return [state], [str(state)], False

        if state is not _STATE_UNSET:
            raise ValueError("state and state_catalog are mutually exclusive")
        if isinstance(state_catalog, (str, bytes, bytearray)) or not isinstance(
            state_catalog,
            Sequence,
        ):
            raise TypeError("state_catalog must be a sequence of state names")
        states = list(state_catalog)
        if not states:
            raise ValueError("state_catalog must contain at least one state")

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
        duplicates = sorted(
            label for label in set(state_labels) if state_labels.count(label) > 1
        )
        if duplicates:
            names = ", ".join(repr(name) for name in duplicates)
            raise ValueError(f"state_catalog contains duplicate states: {names}")
        if any(value in (retro.State.NONE, None) for value in state_values):
            raise ValueError("state_catalog entries must resolve to saved states")
        return state_values, state_labels, True

    @classmethod
    def _resolve_initial_state_payload(
        cls,
        retro,
        game,
        inttype,
        rom_path,
        info_path,
        scenario_path,
        env_kwargs,
        *,
        first_initial_state=_SERIALIZED_UNSET,
        resolved_config=None,
    ):
        if resolved_config is None:
            raise ValueError("resolved state configuration is required")
        state_values, state_labels, explicit_catalog = resolved_config

        if first_initial_state is _SERIALIZED_UNSET:
            first_initial_state = cls._serialize_initial_state(
                retro,
                game,
                state_values[0],
                inttype,
                rom_path,
                info_path,
                scenario_path,
                env_kwargs,
            )

        initial_state = first_initial_state
        initial_state_labels = None
        if explicit_catalog:
            initial_states = [initial_state]
            for value, label in zip(state_values[1:], state_labels[1:]):
                serialized = cls._serialize_initial_state(
                    retro,
                    game,
                    value,
                    inttype,
                    rom_path,
                    info_path,
                    scenario_path,
                    env_kwargs,
                )
                if not serialized:
                    raise ValueError(
                        f"state {label!r} did not resolve to a non-empty state",
                    )
                initial_states.append(serialized)
            if any(not value for value in initial_states):
                raise ValueError("states must resolve to non-empty start states")
            initial_state = initial_states
            initial_state_labels = state_labels
        elif initial_state is not None:
            initial_state_labels = state_labels
        return initial_state, initial_state_labels

    @staticmethod
    def _serialize_initial_state(
        retro,
        game,
        state,
        inttype,
        rom_path,
        info_path,
        scenario_path,
        env_kwargs,
    ):
        template = RetroVecEnv._make_template_env(
            retro,
            game,
            state,
            inttype,
            rom_path,
            info_path,
            scenario_path,
            env_kwargs,
        )
        try:
            return RetroVecEnv._initial_state(template)
        finally:
            template.close()

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
    def _reward_clip_config(reward_clip):
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
        if self._copy_obs:
            return self._observations.copy()
        return self._observations

    def _set_observations(self, observations, operation):
        observations = np.asarray(observations)
        expected_shape = self.observation_space.shape
        if observations.dtype != np.uint8 or observations.shape != expected_shape:
            raise RuntimeError(
                f"native {operation} returned observations with shape "
                f"{observations.shape} and dtype {observations.dtype}; expected "
                f"shape {expected_shape} and dtype uint8",
            )
        self._observations = observations

    def _actions_to_masks(self, actions):
        import stable_retro as retro

        if self.action_mode == "custom_discrete":
            values = np.asarray(actions).reshape(-1)
            masks = np.zeros((self.num_envs, self.num_buttons), dtype=np.uint8)
            for env_idx, value in enumerate(values):
                try:
                    action_bits = self._custom_action_masks[int(value)][0]
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"custom discrete actions must be in [0, {len(self.action_table) - 1}]"
                    ) from exc
                for key in range(self.num_buttons):
                    masks[env_idx, key] = (action_bits >> key) & 1
            return masks
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

    def _reset_seeds(self):
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self):
        self._options = [None for _ in range(self.num_envs)]

    def seed(self, seed: int | None = None):
        """Set per-lane reset seeds for the next ``reset()`` call."""
        if seed is None:
            self._seeds = [None for _ in range(self.num_envs)]
        else:
            base = int(seed)
            self._seeds = [base + i for i in range(self.num_envs)]
        return list(self._seeds)

    def _normalize_reset_seed(self, seed):
        if seed is not None:
            if isinstance(seed, Sequence) and not isinstance(
                seed,
                (str, bytes, bytearray),
            ):
                seeds = [None if value is None else int(value) for value in seed]
                if len(seeds) != self.num_envs:
                    raise ValueError("seed sequence length must match num_envs")
                return seeds
            base = int(seed)
            return [base + i for i in range(self.num_envs)]
        if not self._seeds:
            return None
        seeds = [None if value is None else int(value) for value in self._seeds]
        if all(value is None for value in seeds):
            return None
        return seeds

    def _list_infos_to_dict(self, infos):
        vector_infos = {}
        for env_num, info in enumerate(infos):
            if info:
                vector_infos = self._add_info(vector_infos, dict(info), env_num)
        return vector_infos

    def _step_infos_to_dict(self, infos):
        return self._list_infos_to_dict(infos)

    def reset(self, *, seed: int | Sequence[int | None] | None = None, options=None):
        if self.closed:
            raise RuntimeError("cannot reset a closed environment")
        super().reset(seed=None if isinstance(seed, Sequence) else seed)
        reset_options = {} if options is None else dict(options)
        reset_mask = reset_options.pop("reset_mask", None)
        if reset_mask is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        elif not isinstance(reset_mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a NumPy array")
        elif reset_mask.shape != (self.num_envs,):
            raise ValueError(
                f"options['reset_mask'] must have shape {(self.num_envs,)}"
            )
        elif reset_mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype np.bool_")
        elif not np.any(reset_mask):
            raise ValueError("options['reset_mask'] must select at least one lane")

        snapshots = reset_options.pop("snapshots", None)
        if snapshots is None:
            snapshot_values = [None for _ in range(self.num_envs)]
        else:
            if isinstance(snapshots, (str, bytes, bytearray)) or not isinstance(
                snapshots,
                Sequence,
            ):
                raise TypeError("options['snapshots'] must be a lane-aligned sequence")
            if len(snapshots) != self.num_envs:
                raise ValueError(
                    f"options['snapshots'] must have length {self.num_envs}",
                )
            snapshot_values = list(snapshots)
        snapshot_mask = np.asarray(
            [value is not None for value in snapshot_values],
            dtype=np.bool_,
        )
        if np.any(snapshot_mask & ~reset_mask):
            raise ValueError("snapshots may only be supplied for selected reset lanes")

        state_indices = reset_options.pop("state_indices", None)
        if reset_options:
            names = ", ".join(sorted(reset_options))
            raise ValueError(f"unsupported reset option(s): {names}")
        if state_indices is None:
            default_index = 0 if self.state_catalog else -1
            state_indices = np.full(self.num_envs, default_index, dtype=np.int32)
            state_indices[snapshot_mask] = -1
        elif not isinstance(state_indices, np.ndarray):
            raise TypeError("options['state_indices'] must be a NumPy array")
        elif state_indices.shape != (self.num_envs,):
            raise ValueError(
                f"options['state_indices'] must have shape {(self.num_envs,)}"
            )
        elif state_indices.dtype != np.int32:
            raise TypeError("options['state_indices'] must have dtype np.int32")
        if np.any(state_indices[snapshot_mask] != -1):
            raise ValueError(
                "snapshot reset lanes must use -1 for the static state selector",
            )
        static_mask = reset_mask & ~snapshot_mask
        if self.state_catalog:
            selected = state_indices[static_mask]
            if np.any(selected < 0) or np.any(selected >= len(self.state_catalog)):
                raise ValueError(
                    "selected state_indices entries must index state_catalog",
                )
        elif np.any(state_indices[static_mask] != -1):
            raise ValueError("state_indices require a non-empty state_catalog")

        seeds = self._normalize_reset_seed(seed)
        if seeds is not None and any(
            seeds[index] is not None for index in np.flatnonzero(snapshot_mask)
        ):
            raise ValueError("snapshot reset lanes cannot also specify a seed")
        if snapshots is None:
            obs, infos = self.native.reset(seeds, reset_mask, state_indices)
        else:
            obs, infos = self.native.reset_mixed(
                seeds,
                reset_mask,
                state_indices,
                snapshot_values,
            )
        self._set_observations(obs, "reset")
        if not hasattr(self, "_initialized"):
            self._initialized = np.zeros(self.num_envs, dtype=np.bool_)
        self._initialized[reset_mask] = True
        self._reset_seeds()
        self._reset_options()
        vector_infos = (
            {}
            if getattr(self, "_info_filter_mode", "all") == "none"
            else self._list_infos_to_dict(infos)
        )
        for legacy_key in ("state", "_state", "start_state", "_start_state"):
            vector_infos.pop(legacy_key, None)
        vector_infos["state_index"] = np.array(
            self._active_state_indices,
            dtype=np.int32,
            copy=True,
        )
        vector_infos["_state_index"] = reset_mask.copy()
        start_source = np.full(self.num_envs, "environment", dtype=object)
        start_source[snapshot_mask] = "snapshot"
        vector_infos["start_source"] = start_source
        vector_infos["_start_source"] = reset_mask.copy()
        return self._obs(), vector_infos

    def step(self, actions):
        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        if not np.all(self._initialized):
            raise RuntimeError("all lanes must be reset before the first step")
        masks = self._actions_to_masks(actions)
        obs, rewards, dones, infos = self.native.step(masks)
        self._set_observations(obs, "step")
        terminations = np.array(dones, dtype=bool, copy=True)
        truncations = np.zeros(self.num_envs, dtype=bool)
        return (
            self._obs(),
            np.array(rewards, dtype=np.float32, copy=True),
            terminations,
            truncations,
            {}
            if getattr(self, "_info_filter_mode", "all") == "none"
            else self._step_infos_to_dict(infos),
        )

    def capture_snapshots(self, mask: np.ndarray):
        if self.closed:
            raise RuntimeError("cannot capture snapshots from a closed environment")
        if not self.supports_live_snapshots:
            raise NotImplementedError(
                "live snapshots are unavailable because this core or scenario "
                "cannot preserve exact runtime state",
            )
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"mask must have shape {(self.num_envs,)}")
        if mask.dtype != np.bool_:
            raise TypeError("mask must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("mask must select at least one lane")
        if not np.all(self._initialized[mask]):
            raise RuntimeError("cannot capture a lane before its initial reset")
        return tuple(self.native.capture_snapshots(mask))

    def close(self):
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.closed = True

    def render_lane(self, lane: int):
        if self.closed:
            raise RuntimeError("cannot render a closed environment")
        if isinstance(lane, (bool, np.bool_)):
            raise TypeError("lane must be an integer")
        try:
            lane_index = int(lane)
        except (TypeError, ValueError):
            raise TypeError("lane must be an integer") from None
        if lane_index != lane:
            raise TypeError("lane must be an integer")
        if not 0 <= lane_index < self.num_envs:
            raise IndexError(f"lane must be in [0, {self.num_envs - 1}]")
        return np.asarray(self.native.get_screen(lane_index)).copy()

    def get_images(self):
        if self.render_mode != "rgb_array":
            return [None for _ in range(self.num_envs)]
        return [self.render_lane(lane) for lane in range(self.num_envs)]

    def render(self, mode: str | None = None):
        mode = self.render_mode if mode is None else mode
        if mode == "rgb_array":
            return self.render_lane(0)
        if mode == "human":
            from stable_retro.rendering import SimpleImageViewer

            img = self.native.get_screen(0)
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        raise ValueError(f"unsupported render mode: {mode}")


__all__ = ["RetroVecEnv"]

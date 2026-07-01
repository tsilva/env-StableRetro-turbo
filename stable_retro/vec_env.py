"""Native vector environment for stable-retro rollouts."""

from __future__ import annotations

import json
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


_UNSET = object()


class RetroVecEnv(VecEnv):
    """SB3-compatible native vector env for stable-retro rollouts.

    This is the supported high-throughput path. C++ owns the emulator pool,
    frame skip, preprocessing, frame stacking, autoreset, reward/done
    evaluation, and batched observation buffer.

    ``state`` accepts a single state name, a sequence of one state per env slot,
    or a mapping of state names to positive sampling weights. Mapping weights are
    normalized and sampled independently for each env on every episode reset.

    Native info-transition termination is opt-in and game/config-specific.
    Pass done_on={"name": ("key", "decrease")} to terminate and autoreset
    only lanes whose post-reset baseline changes as requested. Supported ops
    are "change", "increase", and "decrease"; variables may be a string or a
    sequence of strings. Named scenario events can also declare multiple
    trigger objects. Fired rules are reported in info["done_on_info"] with
    list-shaped "keys"/"variables", "prev", and "next" values, including
    one-element lists for single-variable rules.

    obs_copy="safe_view" returns double-buffered observation views so the
    previous observation survives the next step for SB3 rollout collection.
    obs_copy="unsafe_view" restores single-buffer aliasing for benchmarks only.
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
        obs_copy="copy",
        obs_resize=None,
        obs_crop=None,
        obs_grayscale=False,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        frame_skip=1,
        frame_stack=1,
        frame_maxpool=False,
        reset_noops=0,
        action_sticky_prob=0.0,
        reward_clip=False,
        info_filter="all",
        done_on=None,
        copy_observations=_UNSET,
        maxpool_last_two=_UNSET,
        noop_reset_max=_UNSET,
        sticky_action_prob=_UNSET,
        info_mode=_UNSET,
        info_keys=_UNSET,
        done_on_info=_UNSET,
        unsafe_zero_copy=_UNSET,
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

        frame_maxpool = self._resolve_legacy_alias(
            "frame_maxpool",
            frame_maxpool,
            "maxpool_last_two",
            maxpool_last_two,
            False,
        )
        reset_noops = self._resolve_legacy_alias(
            "reset_noops",
            reset_noops,
            "noop_reset_max",
            noop_reset_max,
            0,
        )
        action_sticky_prob = self._resolve_legacy_alias(
            "action_sticky_prob",
            action_sticky_prob,
            "sticky_action_prob",
            sticky_action_prob,
            0.0,
        )
        info_mode, info_keys = self._normalize_info_filter(
            info_filter,
            info_mode,
            info_keys,
        )
        done_on = self._resolve_legacy_alias(
            "done_on",
            done_on,
            "done_on_info",
            done_on_info,
            None,
        )
        copy_observations, unsafe_zero_copy = self._normalize_obs_copy(
            obs_copy,
            copy_observations,
            unsafe_zero_copy,
        )

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
            "maxpool_last_two": frame_maxpool,
            "noop_reset_max": reset_noops,
            "sticky_action_prob": action_sticky_prob,
            "reward_clip": reward_clip,
        }
        info_path = self._resolve_info_path(retro, game, info, inttype)
        scenario_path = self._resolve_scenario_path(retro, game, scenario, inttype)
        done_on_info_rules = self._normalize_done_on(
            done_on,
            label="done_on",
            game=game,
            inttype=inttype,
            scenario_path=scenario_path,
        )
        obs_layout = str(obs_layout).lower()
        if obs_layout not in {"hwc", "chw"}:
            raise ValueError("obs_layout must be 'hwc' or 'chw'")
        self.obs_layout = obs_layout
        self.obs_copy = (
            "unsafe_view"
            if unsafe_zero_copy
            else "copy" if copy_observations else "safe_view"
        )
        self.copy_observations = copy_observations
        self.unsafe_zero_copy = unsafe_zero_copy
        self.render_mode = render_mode
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
    def _resolve_legacy_alias(new_name, new_value, old_name, old_value, default):
        if old_value is _UNSET:
            return new_value
        if new_value != default:
            raise ValueError(f"cannot pass both {new_name} and {old_name}")
        return old_value

    @staticmethod
    def _normalize_obs_copy(obs_copy, copy_observations, unsafe_zero_copy):
        if copy_observations is not _UNSET or unsafe_zero_copy is not _UNSET:
            if obs_copy != "copy":
                raise ValueError(
                    "cannot pass both obs_copy and copy_observations/unsafe_zero_copy",
                )
            legacy_copy = (
                True
                if copy_observations is _UNSET
                else bool(copy_observations)
            )
            legacy_unsafe = (
                False
                if unsafe_zero_copy is _UNSET
                else bool(unsafe_zero_copy)
            )
            if legacy_copy and legacy_unsafe:
                raise ValueError(
                    "unsafe_zero_copy=True is only valid with copy_observations=False",
                )
            return legacy_copy, legacy_unsafe

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

    @classmethod
    def _normalize_info_filter(cls, info_filter, info_mode, info_keys):
        if info_mode is not _UNSET or info_keys is not _UNSET:
            if info_filter != "all":
                raise ValueError(
                    "cannot pass both info_filter and info_mode/info_keys",
                )
            mode = "all" if info_mode is _UNSET else str(info_mode)
            keys = None if info_keys is _UNSET else info_keys
            return mode, cls._normalize_info_keys(keys)

        if info_filter is None:
            return "all", None
        if isinstance(info_filter, str):
            return str(info_filter), None
        if not isinstance(info_filter, Mapping):
            raise ValueError(
                "info_filter must be a mode string or a mapping with mode/keys",
            )
        unknown = set(info_filter) - {"mode", "keys"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"unknown info_filter keys: {names}")
        mode = str(info_filter.get("mode", "all"))
        keys = cls._normalize_info_keys(info_filter.get("keys", None))
        return mode, keys

    @staticmethod
    def _normalize_info_keys(info_keys):
        if isinstance(info_keys, str):
            raise ValueError(
                "info_filter keys must be a sequence of strings, not a string",
            )
        if info_keys is None:
            return None
        return [str(key) for key in info_keys]

    @classmethod
    def _normalize_done_on(
        cls,
        done_on,
        *,
        label,
        game=None,
        inttype=None,
        scenario_path=None,
    ):
        rules = []
        if done_on is not None:
            if isinstance(done_on, Sequence) and not isinstance(
                done_on,
                (str, bytes, bytearray),
            ):
                done_on = {str(name): None for name in done_on}
            if not isinstance(done_on, Mapping):
                raise ValueError(
                    f"{label} must be a mapping of rule names to event specs "
                    "or a sequence of configured event names",
                )
            for raw_name, spec in done_on.items():
                name = str(raw_name)
                if not name:
                    raise ValueError(f"{label} rule names must not be empty")
                if spec is None:
                    spec = cls.resolve_info_event_rules(
                        game,
                        (name,),
                        inttype=inttype,
                        label=label,
                        scenario_path=scenario_path,
                    )[name]
                for trigger_id, variables, op, compare in cls._normalize_event_spec(
                    name,
                    spec,
                    label=label,
                ):
                    rules.append((name, trigger_id, variables, op, compare))
        if not rules:
            return None
        return tuple(rules)

    @classmethod
    def _normalize_event_spec(cls, name, spec, *, label):
        if cls._is_compact_event_spec(spec):
            return (cls._normalize_event_trigger(name, spec, label=label),)

        if not isinstance(spec, Mapping):
            raise ValueError(
                f"{label} values must be compact (variables, op) pairs or "
                "mappings with variables/op or triggers",
            )

        allowed = {"description", "triggers", "id", "variables", "keys", "op", "compare"}
        unknown = set(spec) - allowed
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"{label} event {name!r} has unknown keys: {names}")

        if "triggers" not in spec:
            return (cls._normalize_event_trigger(name, spec, label=label),)

        raw_triggers = spec["triggers"]
        if isinstance(raw_triggers, (str, bytes, bytearray)) or not isinstance(
            raw_triggers,
            Sequence,
        ):
            raise ValueError(f"{label} event {name!r} triggers must be a sequence")
        if not raw_triggers:
            raise ValueError(f"{label} event {name!r} must contain at least one trigger")

        triggers = []
        trigger_count = len(raw_triggers)
        for index, raw_trigger in enumerate(raw_triggers):
            triggers.append(
                cls._normalize_event_trigger(
                    name,
                    raw_trigger,
                    label=label,
                    index=index,
                    trigger_count=trigger_count,
                ),
            )
        return tuple(triggers)

    @staticmethod
    def _is_compact_event_spec(spec):
        return (
            isinstance(spec, Sequence)
            and not isinstance(spec, (str, bytes, bytearray))
            and len(spec) == 2
        )

    @classmethod
    def _normalize_event_trigger(
        cls,
        event_name,
        trigger,
        *,
        label,
        index=0,
        trigger_count=1,
    ):
        if cls._is_compact_event_spec(trigger):
            raw_variables, raw_op = trigger
            trigger_id = "default" if trigger_count == 1 else f"trigger_{index + 1}"
            compare = "reset"
        elif isinstance(trigger, Mapping):
            allowed = {"description", "id", "variables", "keys", "op", "compare"}
            unknown = set(trigger) - allowed
            if unknown:
                names = ", ".join(sorted(str(key) for key in unknown))
                raise ValueError(
                    f"{label} event {event_name!r} trigger has unknown keys: {names}",
                )
            if "variables" in trigger and "keys" in trigger:
                raise ValueError(
                    f"{label} event {event_name!r} trigger cannot use both "
                    "variables and keys",
                )
            raw_variables = trigger.get("variables", trigger.get("keys"))
            raw_op = trigger.get("op")
            trigger_id = str(
                trigger.get(
                    "id",
                    "default" if trigger_count == 1 else f"trigger_{index + 1}",
                ),
            )
            compare = str(trigger.get("compare", "reset"))
        else:
            raise ValueError(
                f"{label} event {event_name!r} triggers must be compact pairs "
                "or mappings",
            )

        if not trigger_id:
            raise ValueError(f"{label} event {event_name!r} trigger ids must not be empty")
        variables = cls._normalize_event_variables(raw_variables, label=label)
        op = cls._normalize_event_op(raw_op, label=label)
        compare = cls._normalize_event_compare(compare, label=label)
        return trigger_id, variables, op, compare

    @staticmethod
    def _normalize_event_variables(raw_variables, *, label):
        if isinstance(raw_variables, str):
            variables = (raw_variables,)
        elif (
            isinstance(raw_variables, Sequence)
            and not isinstance(raw_variables, (bytes, bytearray))
        ):
            variables = tuple(str(variable) for variable in raw_variables)
        else:
            raise ValueError(
                f"{label} variables must be a string or sequence of strings",
            )
        if not variables or any(not variable for variable in variables):
            raise ValueError(f"{label} rules must reference at least one variable")
        return variables

    @staticmethod
    def _normalize_event_op(raw_op, *, label):
        op = str(raw_op)
        if op not in {"change", "increase", "decrease"}:
            raise ValueError(
                f"{label} ops must be 'change', 'increase', or 'decrease'",
            )
        return op

    @staticmethod
    def _normalize_event_compare(raw_compare, *, label):
        compare = str(raw_compare)
        if compare != "reset":
            raise ValueError(f"{label} compare must be 'reset'")
        return compare

    @staticmethod
    def _normalize_done_on_info(done_on_info):
        return RetroVecEnv._normalize_done_on(done_on_info, label="done_on_info")

    @staticmethod
    def metadata_info_events(game, inttype=None):
        """Return named info-event rules declared by a game's metadata.

        Kept as a compatibility fallback for versions that stored events there.
        New game integrations should prefer scenario ``events``.
        """

        if not game:
            return {}
        if inttype is None:
            inttype = retro_data.Integrations.STABLE
        try:
            metadata_path = retro_data.get_file_path(game, "metadata.json", inttype)
        except FileNotFoundError:
            return {}
        if not metadata_path:
            return {}
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        info_events = metadata.get("info_events", {})
        return info_events if isinstance(info_events, Mapping) else {}

    @staticmethod
    def scenario_events(scenario_path):
        """Return named event rules declared by a scenario file."""

        if not scenario_path:
            return {}
        try:
            with open(scenario_path, encoding="utf-8") as handle:
                scenario = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        events = scenario.get("events", {})
        return events if isinstance(events, Mapping) else {}

    @classmethod
    def resolve_info_event_rules(
        cls,
        game,
        names,
        *,
        inttype=None,
        label="info_events",
        scenario_path=None,
    ):
        """Resolve configured event names to raw done_on rule specs."""

        if not game and not scenario_path:
            raise ValueError(f"{label} named events require a game or scenario")
        event_rules = {}
        if game:
            event_rules.update(cls.metadata_info_events(game, inttype=inttype))
        event_rules.update(cls.scenario_events(scenario_path))
        resolved = {}
        missing = []
        for raw_name in names:
            name = str(raw_name)
            if not name:
                raise ValueError(f"{label} event names must not be empty")
            if name not in event_rules:
                missing.append(name)
                continue
            resolved[name] = event_rules[name]
        if missing:
            available = ", ".join(sorted(str(name) for name in event_rules)) or "none"
            source = game or str(scenario_path)
            raise ValueError(
                f"{label} unknown configured event(s) for {source}: "
                f"{', '.join(missing)}. Available events: {available}",
            )
        return resolved

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

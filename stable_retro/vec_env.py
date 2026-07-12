"""Gymnasium vector environments for stable-retro rollouts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import stable_retro.data as retro_data
from stable_retro.enums import Actions, Observations, State
from stable_retro.retro_env import RetroEnv

_SERIALIZED_UNSET = object()


class RetroVecEnv(VectorEnv):
    """Gymnasium vector env for stable-retro rollouts.

    This is the supported high-throughput path. C++ owns the emulator pool,
    frame skip, preprocessing, frame stacking, reward/done evaluation, and
    batched observation buffer.

    ``autoreset_mode`` defaults to Gymnasium same-step autoreset for backwards
    compatibility. With ``AutoresetMode.DISABLED``, terminated lanes keep their
    terminal observation and cannot be stepped again until selected with
    ``reset(options={"reset_mask": mask})``. Masked reset leaves every
    unselected lane untouched.

    ``state`` accepts a single state name, a sequence of one state per env slot,
    or a mapping of state names to non-negative sampling weights. Mapping weights
    are normalized and sampled independently for each env on every episode reset.
    Use ``set_state_policy()`` with the same shapes to update the reset policy
    used by future resets and autoresets.

    On Stella-backed Atari environments, ``use_fire_reset=True`` presses FIRE
    for one native frame after each full-episode reset when FIRE is available,
    then releases it before reset no-ops. This matches ALE's vector-env default.

    Native info-transition termination is opt-in and game/config-specific.
    Pass done_on={"name": ("key", "decrease")} to terminate only lanes whose
    post-reset baseline changes as requested. In same-step mode those lanes are
    autoreset; in disabled mode they remain terminal until explicitly reset.
    Supported ops are "change", "increase", and "decrease"; variables may be a
    sequence of strings. Named scenario events can also declare multiple
    trigger objects. Fired rules are reported in info["done_on_info"] with
    list-shaped "keys"/"variables", "prev", and "next" values, including
    one-element lists for single-variable rules.

    obs_copy="safe_view" returns double-buffered observation views so the
    previous observation survives the next step for rollout collection.
    obs_copy="unsafe_view" restores single-buffer aliasing for benchmarks only.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

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
        obs_crop_mode: Literal["remove", "mask"] = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale=False,
        obs_resize_algorithm="nearest",
        obs_layout="hwc",
        frame_skip=1,
        frame_stack=1,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=True,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="all",
        done_on=None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.SAME_STEP,
    ):
        import stable_retro as retro
        from stable_retro import _retro

        num_envs = RetroEnv._normalize_positive_int(num_envs, "num_envs")
        if num_threads is not None:
            num_threads = RetroEnv._normalize_positive_int(num_threads, "num_threads")
        obs_resize = RetroEnv._normalize_obs_resize(obs_resize)
        obs_crop = RetroEnv._normalize_obs_crop(obs_crop)
        obs_crop_fill = RetroEnv._normalize_obs_crop_fill(obs_crop_fill)
        obs_resize_algorithm = RetroEnv._normalize_obs_resize_algorithm(
            obs_resize_algorithm,
        )
        frame_skip = RetroEnv._normalize_positive_int(frame_skip, "frame_skip")
        frame_stack = RetroEnv._normalize_positive_int(frame_stack, "frame_stack")
        noop_reset_max = RetroEnv._normalize_nonnegative_int(
            noop_reset_max,
            "noop_reset_max",
        )
        use_fire_reset = bool(use_fire_reset)
        sticky_action_prob = RetroEnv._normalize_probability(
            sticky_action_prob,
            "sticky_action_prob",
        )
        reward_clip = RetroEnv._normalize_reward_clip(reward_clip)
        autoreset_mode = self._normalize_autoreset_mode(autoreset_mode)

        self.closed = False
        self.autoreset_mode = autoreset_mode
        self.metadata = dict(type(self).metadata)
        self.metadata["autoreset_mode"] = autoreset_mode
        self._observations = None
        self._seeds = [None for _ in range(num_envs)]
        self._options = [None for _ in range(num_envs)]
        self._game = game
        self._inttype = inttype
        self._rom_path = rom_path
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
            "obs_resize": obs_resize,
            "obs_crop": obs_crop,
            "obs_crop_mode": obs_crop_mode,
            "obs_crop_fill": obs_crop_fill,
            "obs_grayscale": obs_grayscale,
            "obs_resize_algorithm": obs_resize_algorithm,
            "frame_skip": frame_skip,
            "frame_stack": frame_stack,
            "maxpool_last_two": maxpool_last_two,
            "noop_reset_max": noop_reset_max,
            "sticky_action_prob": sticky_action_prob,
            "reward_clip": reward_clip,
        }
        info_path = self._resolve_info_path(retro, game, info, inttype)
        scenario_path = self._resolve_scenario_path(retro, game, scenario, inttype)
        self._info_path = info_path
        self._scenario_path = scenario_path
        self._env_kwargs = env_kwargs
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
        done_on_rules = self._normalize_done_on(
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
            if unsafe_view
            else "copy" if copy_obs else "safe_view"
        )
        self._copy_obs = copy_obs
        self._unsafe_view = unsafe_view
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
            crop = template._effective_crop(0, height, width)
            crop_mask = template._native_mask_crop()
            initial_state = template.initial_state if template.initial_state else None
            self.single_action_space = template.action_space
            self.single_observation_space = self._observation_space_for_layout(
                template.observation_space,
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
            self.use_restricted_actions = template.use_restricted_actions
            self._filter_actions = self.use_restricted_actions == retro.Actions.FILTERED
            reward_clip, reward_low, reward_high = self._reward_clip_config(template)
        finally:
            template.close()

        initial_state, initial_state_labels, initial_state_weights = (
            self._resolve_initial_state_payload(
                retro,
                game,
                num_envs,
                state,
                inttype,
                rom_path,
                info_path,
                scenario_path,
                env_kwargs,
                first_initial_state=initial_state,
                resolved_config=(
                    state_values,
                    state_labels,
                    state_probs,
                    state_collection,
                ),
            )
        )

        resolved_rom_path = rom_path or retro.data.get_original_romfile_path(
            game,
            inttype,
        )
        if num_threads is None:
            num_threads = num_envs
        self.native = _retro._RetroVecEnv(
            num_envs,
            str(resolved_rom_path),
            str(info_path),
            str(scenario_path),
            initial_state,
            int(self.num_buttons),
            frame_skip,
            frame_stack,
            crop,
            obs_resize,
            bool(obs_grayscale),
            str(env_kwargs.get("obs_resize_algorithm", "nearest")),
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
            done_on_rules,
            initial_state_labels,
            initial_state_weights,
            crop_mask,
            obs_crop_fill,
            autoreset_mode is AutoresetMode.SAME_STEP,
        )
        self.num_envs = num_envs
        self.initial_state_names = tuple(self.native.initial_state_names)
        self._active_state_indices = self.native.active_state_indices()
        self._active_state_indices.setflags(write=False)
        self._active_state_names = [None for _ in range(self.num_envs)]

    @staticmethod
    def _fire_reset_button(template, enabled):
        """Return the Stella FIRE button index, or -1 when unavailable."""
        if not enabled or template.system != "Atari2600":
            return -1
        try:
            fire_button = template.buttons.index("BUTTON")
        except ValueError:
            return -1
        if template.use_restricted_actions == Actions.ALL:
            return fire_button
        fire_mask = 1 << fire_button
        if any(
            int(action) & fire_mask
            for combo in template.button_combos
            for action in combo
        ):
            return fire_button
        return -1

    @staticmethod
    def _normalize_autoreset_mode(value):
        if isinstance(value, AutoresetMode):
            mode = value
        else:
            try:
                mode = AutoresetMode(value)
            except (TypeError, ValueError):
                name = str(value).split(".")[-1].upper()
                try:
                    mode = AutoresetMode[name]
                except KeyError as exc:
                    raise ValueError(f"unsupported autoreset_mode: {value!r}") from exc
        if mode not in (AutoresetMode.SAME_STEP, AutoresetMode.DISABLED):
            raise ValueError(
                "autoreset_mode must be AutoresetMode.SAME_STEP or AutoresetMode.DISABLED",
            )
        return mode

    def set_state_policy(self, state):
        """Update the reset policy used at future episode boundaries.

        Accepts the same shapes as the constructor ``state`` argument: a single
        state, a per-lane sequence, or a weighted mapping. Currently active lanes
        are not interrupted; the new policy is used by the next explicit reset or
        per-lane autoreset.
        """
        import stable_retro as retro

        initial_state, initial_state_labels, initial_state_weights = (
            self._resolve_initial_state_payload(
                retro,
                self._game,
                self.num_envs,
                state,
                self._inttype,
                self._rom_path,
                self._info_path,
                self._scenario_path,
                self._env_kwargs,
            )
        )
        self.native.set_initial_states(
            initial_state,
            initial_state_labels,
            initial_state_weights,
        )
        self.initial_state_names = tuple(self.native.initial_state_names)

    def set_state_sampling_weights(self, weights):
        """Compatibility alias for updating a weighted state reset policy."""
        if isinstance(weights, Mapping):
            self.set_state_policy(weights)
            return
        self.set_state_policy(
            dict(zip(self.native.initial_state_policy_names(), weights, strict=True)),
        )

    def state_sampling_weights(self):
        """Return current normalized reset-sampling weights by state name."""
        return dict(
            zip(
                self.native.initial_state_policy_names(),
                self.native.initial_state_weights(),
                strict=True,
            ),
        )

    def active_state_indices(self):
        """Return a read-only int32 NumPy view of active initial-state indices.

        The returned array is owned by this env and mutates in place
        after ``reset()`` and after per-lane automatic resets inside
        ``step()``. Copy it when a stable snapshot is needed.
        Lanes without a serialized initial state report ``-1``.
        """
        return self._active_state_indices

    def active_states(self):
        """Return active initial-state names for each lane."""
        return tuple(self._active_state_names)

    def _refresh_active_state_names(self, lanes=None):
        if not hasattr(self, "initial_state_names"):
            return
        if not hasattr(self, "_active_state_names"):
            self._active_state_names = [None for _ in range(self.num_envs)]
        names = self.initial_state_names
        if lanes is None:
            lanes = range(self.num_envs)
        for lane in lanes:
            index = int(self._active_state_indices[lane])
            self._active_state_names[lane] = None if index < 0 else names[index]

    @staticmethod
    def _normalize_state_sampling_weights(weights, state_names):
        if not state_names:
            raise ValueError("state sampling weights require named initial states")

        if isinstance(weights, Mapping):
            unknown = set(weights) - set(state_names)
            missing = set(state_names) - set(weights)
            if unknown:
                names = ", ".join(sorted(str(name) for name in unknown))
                raise ValueError(f"unknown state sampling weight names: {names}")
            if missing:
                names = ", ".join(sorted(str(name) for name in missing))
                raise ValueError(f"missing state sampling weights: {names}")
            raw_weights = [weights[name] for name in state_names]
        elif isinstance(weights, Sequence) and not isinstance(
            weights,
            (str, bytes, bytearray),
        ):
            raw_weights = list(weights)
            if len(raw_weights) != len(state_names):
                raise ValueError(
                    "state sampling weight sequence length must match initial_state_names",
                )
        else:
            raise ValueError(
                "state sampling weights must be a mapping or a sequence",
            )

        normalized_weights = []
        for weight in raw_weights:
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "state sampling weights must contain non-negative finite numbers",
                )
            normalized_weights.append(value)
        total = math.fsum(normalized_weights)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("state sampling weights must sum to a positive number")
        return [value / total for value in normalized_weights]

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

        if not scenario_path:
            raise ValueError(f"{label} named events require a scenario")
        event_rules = cls.scenario_events(scenario_path)
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
            raise ValueError(
                f"{label} unknown configured event(s) for {scenario_path}: "
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
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "state weights must contain non-negative finite numbers",
                )
            normalized_probs.append(value)
        total = math.fsum(normalized_probs)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("state weights must sum to a positive number")
        normalized_probs = [value / total for value in normalized_probs]
        return state_values, state_labels, normalized_probs, state_collection

    @classmethod
    def _resolve_initial_state_payload(
        cls,
        retro,
        game,
        num_envs,
        state,
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
            resolved_config = cls._resolve_state_config(retro, game, num_envs, state)
        state_values, state_labels, state_probs, state_collection = resolved_config

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
        initial_state_weights = None
        if state_collection:
            initial_states = [initial_state]
            state_cache = {state_labels[0]: initial_state}
            for value, label in zip(state_values[1:], state_labels[1:]):
                cached = state_cache.get(label)
                if cached is not None:
                    initial_states.append(cached)
                    continue
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
                state_cache[label] = serialized
                initial_states.append(serialized)
            if any(not value for value in initial_states):
                raise ValueError("states must resolve to non-empty start states")
            initial_state = initial_states
            initial_state_labels = state_labels
            initial_state_weights = state_probs
        elif initial_state is not None:
            initial_state_labels = state_labels
        return initial_state, initial_state_labels, initial_state_weights

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
            return template.initial_state if template.initial_state else None
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
        if self._copy_obs:
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
        vector_infos = {}
        for env_num, raw_info in enumerate(infos):
            info = dict(raw_info)
            terminal_observation = info.pop("terminal_observation", None)
            reset_info = info.pop("reset_info", None)
            terminal_info = info.pop("terminal_info", None)
            info.pop("TimeLimit.truncated", None)

            if terminal_observation is not None:
                final_info = dict(terminal_info or info)
                done_on_info = info.get("done_on_info")
                if done_on_info is not None:
                    final_info["done_on_info"] = done_on_info
                if reset_info:
                    info.update(dict(reset_info))
                info["final_obs"] = terminal_observation
                info["final_info"] = final_info

            if info:
                vector_infos = self._add_info(vector_infos, info, env_num)
        return vector_infos

    def reset(self, *, seed: int | Sequence[int | None] | None = None, options=None):
        super().reset(seed=None if isinstance(seed, Sequence) else seed)
        reset_options = {} if options is None else dict(options)
        reset_mask = reset_options.pop("reset_mask", None)
        if reset_mask is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        elif not isinstance(reset_mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a NumPy array")
        elif reset_mask.shape != (self.num_envs,):
            raise ValueError(f"options['reset_mask'] must have shape {(self.num_envs,)}")
        elif reset_mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype np.bool_")
        elif not np.any(reset_mask):
            raise ValueError("options['reset_mask'] must select at least one lane")

        start_indices = reset_options.pop("start_indices", None)
        if reset_options:
            names = ", ".join(sorted(reset_options))
            raise ValueError(f"unsupported reset option(s): {names}")
        if start_indices is None:
            start_indices = np.full(self.num_envs, -1, dtype=np.int32)
        elif not isinstance(start_indices, np.ndarray):
            raise TypeError("options['start_indices'] must be a NumPy array")
        elif start_indices.shape != (self.num_envs,):
            raise ValueError(f"options['start_indices'] must have shape {(self.num_envs,)}")
        elif start_indices.dtype != np.int32:
            raise TypeError("options['start_indices'] must have dtype np.int32")

        seeds = self._normalize_reset_seed(seed)
        obs, infos = self.native.reset(seeds, reset_mask, start_indices)
        self._observations = np.asarray(obs, dtype=np.uint8)
        self._refresh_active_state_names(np.flatnonzero(reset_mask))
        self._reset_seeds()
        self._reset_options()
        vector_infos = (
            {}
            if getattr(self, "_info_filter_mode", "all") == "none"
            else self._list_infos_to_dict(infos)
        )
        return self._obs(), vector_infos

    def step(self, actions):
        masks = self._actions_to_masks(actions)
        obs, rewards, dones, infos = self.native.step(masks)
        self._observations = np.asarray(obs, dtype=np.uint8)
        terminations = np.array(dones, dtype=bool, copy=True)
        if np.any(terminations):
            self._refresh_active_state_names(np.flatnonzero(terminations))
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

    def close(self):
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.closed = True

    def get_images(self):
        return []

    def render(self, mode: str | None = None):
        mode = self.render_mode if mode is None else mode
        if mode == "rgb_array":
            return self.native.get_screen(0)
        if mode == "human":
            from stable_retro.rendering import SimpleImageViewer

            img = self.native.get_screen(0)
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        raise ValueError(f"unsupported render mode: {mode}")


__all__ = ["RetroVecEnv"]

"""Native vector environment for stable-retro rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover - import remains cheap without SB3.
    VecEnv = object


class StableRetroNativeVecEnv(VecEnv):
    """SB3-compatible native vector env for homogeneous stable-retro rollouts.

    This is the supported high-throughput path. C++ owns the emulator pool,
    frame skip, preprocessing, frame stacking, autoreset, reward/done
    evaluation, and batched observation buffer.
    """

    def __init__(
        self,
        game,
        num_envs: int,
        state=None,
        inttype=None,
        rom_path: str | None = None,
        info=None,
        scenario=None,
        num_threads: int | None = None,
        copy_observations: bool = True,
        **env_kwargs,
    ):
        if VecEnv is object:
            raise ImportError(
                "StableRetroNativeVecEnv requires stable-baselines3 to be installed",
            )
        import stable_retro as retro
        from stable_retro import _retro

        if state is None:
            state = retro.State.DEFAULT
        if inttype is None:
            inttype = retro.data.Integrations.DEFAULT
        self.copy_observations = bool(copy_observations)
        self.waiting = False
        self.closed = False
        self._actions = None
        self._observations = None

        env_kwargs = dict(env_kwargs)
        env_kwargs.setdefault("render_mode", "rgb_array")
        self.render_mode = env_kwargs["render_mode"]
        if env_kwargs.get("players", 1) != 1:
            raise ValueError("StableRetroNativeVecEnv currently supports players=1")
        if env_kwargs.get("record", False):
            raise ValueError("StableRetroNativeVecEnv does not support movie recording")
        if (
            env_kwargs.get("obs_type", retro.Observations.IMAGE)
            != retro.Observations.IMAGE
        ):
            raise ValueError(
                "StableRetroNativeVecEnv currently supports image observations only",
            )

        info_path = self._resolve_info_path(retro, game, info, inttype)
        scenario_path = self._resolve_scenario_path(retro, game, scenario, inttype)
        template = self._make_template_env(
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
            if template._rotation_steps() != 0:
                raise ValueError(
                    "StableRetroNativeVecEnv does not support rotated screens",
                )
            width, height = template.em.get_resolution()
            crop = template._effective_crop(0, height, width)
            initial_state = template.initial_state if template.initial_state else None
            self.action_space = template.action_space
            self.observation_space = template.observation_space
            self.num_buttons = template.num_buttons
            self.button_combos = [
                [int(action) for action in combo] for combo in template.button_combos
            ]
            self.use_restricted_actions = template.use_restricted_actions
            self._filter_actions = self.use_restricted_actions == retro.Actions.FILTERED
            reward_clip, reward_low, reward_high = self._reward_clip_config(template)
        finally:
            template.close()

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
        )
        super().__init__(int(num_envs), self.observation_space, self.action_space)

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
        seed = None
        if self._seeds:
            seeds = [value for value in self._seeds if value is not None]
            if seeds:
                seed = int(seeds[0])
        obs, infos = self.native.reset(seed)
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
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
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


__all__ = ["StableRetroNativeVecEnv"]

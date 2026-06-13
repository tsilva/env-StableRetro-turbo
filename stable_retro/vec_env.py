"""Vector environments for stable-retro workers."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import cloudpickle
except ImportError:  # pragma: no cover - only used when SB3 is absent too.
    cloudpickle = None

try:
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover - import remains cheap without SB3.
    VecEnv = object


class CloudpickleWrapper:
    """Serialize callables with cloudpickle when multiprocessing uses spawn."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        if cloudpickle is None:
            return self.fn
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state):
        if cloudpickle is None:
            self.fn = state
        else:
            self.fn = cloudpickle.loads(state)

    def __call__(self):
        return self.fn()


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    shm = None
    obs_array = None
    obs_index = None
    try:
        remote.send((env.observation_space, env.action_space))
        while True:
            cmd, data = remote.recv()
            if cmd == "set_shm":
                name, shape, dtype, obs_index = data
                shm = shared_memory.SharedMemory(name=name)
                obs_array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                remote.send(None)
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                if done:
                    terminal_observation = obs
                    obs, reset_info = env.reset()
                    info = dict(info)
                    info["terminal_observation"] = terminal_observation
                    info["reset_info"] = reset_info
                    info["TimeLimit.truncated"] = bool(truncated and not terminated)
                obs_array[obs_index] = obs
                remote.send((reward, done, info))
            elif cmd == "reset":
                seed, options = data
                kwargs = {}
                if seed is not None:
                    kwargs["seed"] = seed
                if options is not None:
                    kwargs["options"] = options
                obs, info = env.reset(**kwargs)
                obs_array[obs_index] = obs
                remote.send(info)
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                name, value = data
                setattr(env, name, value)
                remote.send(None)
            elif cmd == "env_method":
                method_name, method_args, method_kwargs = data
                remote.send(getattr(env, method_name)(*method_args, **method_kwargs))
            elif cmd == "is_wrapped":
                wrapper_class = data
                current = env
                wrapped = False
                while isinstance(current, gym.Wrapper):
                    if isinstance(current, wrapper_class):
                        wrapped = True
                        break
                    current = current.env
                remote.send(wrapped)
            else:
                raise NotImplementedError(cmd)
    finally:
        if shm is not None:
            shm.close()


def _chunk_worker(remote, parent_remote, env_fn_wrappers):
    parent_remote.close()
    try:
        envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers]
    except Exception as exc:
        remote.send(("error", f"{type(exc).__name__}: {exc}"))
        remote.close()
        return
    shm = None
    obs_array = None
    obs_indices = None
    try:
        remote.send(
            (
                "spaces",
                [env.observation_space for env in envs],
                [env.action_space for env in envs],
            ),
        )
        while True:
            cmd, data = remote.recv()
            if cmd == "set_shm":
                name, shape, dtype, obs_indices = data
                shm = shared_memory.SharedMemory(name=name)
                obs_array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                remote.send(None)
            elif cmd == "step":
                rewards = []
                dones = []
                infos = []
                for env, action, obs_index in zip(envs, data, obs_indices):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    if done:
                        terminal_observation = obs
                        obs, reset_info = env.reset()
                        info = dict(info)
                        info["terminal_observation"] = terminal_observation
                        info["reset_info"] = reset_info
                        info["TimeLimit.truncated"] = bool(truncated and not terminated)
                    obs_array[obs_index] = obs
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                remote.send((rewards, dones, infos))
            elif cmd == "reset":
                infos = []
                for env, (seed, options), obs_index in zip(envs, data, obs_indices):
                    kwargs = {}
                    if seed is not None:
                        kwargs["seed"] = seed
                    if options is not None:
                        kwargs["options"] = options
                    obs, info = env.reset(**kwargs)
                    obs_array[obs_index] = obs
                    infos.append(info)
                remote.send(infos)
            elif cmd == "render":
                remote.send([env.render() for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_attr":
                attr_name, local_indices = data
                remote.send([getattr(envs[i], attr_name) for i in local_indices])
            elif cmd == "set_attr":
                name, value, local_indices = data
                for i in local_indices:
                    setattr(envs[i], name, value)
                remote.send(None)
            elif cmd == "env_method":
                method_name, method_args, method_kwargs, local_indices = data
                remote.send(
                    [
                        getattr(envs[i], method_name)(*method_args, **method_kwargs)
                        for i in local_indices
                    ],
                )
            elif cmd == "is_wrapped":
                wrapper_class, local_indices = data
                results = []
                for i in local_indices:
                    current = envs[i]
                    wrapped = False
                    while isinstance(current, gym.Wrapper):
                        if isinstance(current, wrapper_class):
                            wrapped = True
                            break
                        current = current.env
                    results.append(wrapped)
                remote.send(results)
            else:
                raise NotImplementedError(cmd)
    finally:
        if shm is not None:
            shm.close()


def _chunk_env_fns(env_fns, chunk_size):
    return [env_fns[i : i + chunk_size] for i in range(0, len(env_fns), chunk_size)]


def _thread_step(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    if done:
        terminal_observation = obs
        obs, reset_info = env.reset()
        info = dict(info)
        info["terminal_observation"] = terminal_observation
        info["reset_info"] = reset_info
        info["TimeLimit.truncated"] = bool(truncated and not terminated)
    return obs, reward, done, info


class StableRetroThreadedVecEnv(VecEnv):
    """SB3-compatible same-process threaded vector env for stable-retro games.

    This env relies on the native frontend supporting multiple in-process
    emulator instances. It avoids subprocess IPC and runs env-local fused native
    stepping in Python threads; the expensive C++ frame-repeat path releases the
    GIL while emulator frames advance.
    """

    def __init__(
        self,
        env_fns,
        num_threads: int | None = None,
        copy_observations: bool = True,
        use_native_batch: bool | None = None,
    ):
        if VecEnv is object:
            raise ImportError(
                "StableRetroThreadedVecEnv requires stable-baselines3 to be installed",
            )
        self.waiting = False
        self.closed = False
        self.copy_observations = bool(copy_observations)
        self.envs = [env_fn() for env_fn in env_fns]
        n_envs = len(self.envs)
        if n_envs <= 0:
            raise ValueError("env_fns must contain at least one environment factory")
        if num_threads is None:
            num_threads = n_envs
        self.num_threads = max(1, int(num_threads))
        if use_native_batch is None:
            use_native_batch = (
                os.environ.get("STABLE_RETRO_DISABLE_NATIVE_BATCH_STEP") != "1"
            )
        self.use_native_batch = bool(use_native_batch)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self._futures = None
        self._actions = None
        self._observations = None

        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space
        for env in self.envs[1:]:
            if env.observation_space != observation_space:
                raise ValueError("all envs must have the same observation space")
            if env.action_space != action_space:
                raise ValueError("all envs must have the same action space")

        super().__init__(n_envs, observation_space, action_space)

    def _can_use_native_batch(self):
        import stable_retro as retro

        if not self.envs:
            return False
        if any(not hasattr(env, "em") or not hasattr(env, "data") for env in self.envs):
            return False
        first = self.envs[0]
        first_config = (
            first._frame_skip,
            first._obs_resize,
            first._obs_grayscale,
            first._obs_resize_algorithm,
            first._maxpool_last_two,
        )
        for env in self.envs:
            config = (
                env._frame_skip,
                env._obs_resize,
                env._obs_grayscale,
                env._obs_resize_algorithm,
                env._maxpool_last_two,
            )
            if (
                env.players != 1
                or env.movie
                or env._obs_type != retro.Observations.IMAGE
                or env._rotation_steps() != 0
                or config != first_config
                or not hasattr(
                    getattr(env.em, "native_emulator", env.em),
                    "get_resolution",
                )
            ):
                return False
        return True

    def _obs(self):
        if self.copy_observations:
            return self._observations.copy()
        return self._observations

    def reset(self):
        futures = []
        for env_idx, env in enumerate(self.envs):
            seed = self._seeds[env_idx] if self._seeds else None
            options = self._options[env_idx] if self._options else None
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed
            if options is not None:
                kwargs["options"] = options
            futures.append(self.executor.submit(env.reset, **kwargs))
        results = [future.result() for future in futures]
        observations, infos = zip(*results)
        self._observations = np.stack(observations)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        return self._obs()

    def step_async(self, actions):
        if self.use_native_batch and self._can_use_native_batch():
            self._actions = np.asarray(actions)
            self._futures = None
            self.waiting = True
            return
        self._futures = [
            self.executor.submit(_thread_step, env, action)
            for env, action in zip(self.envs, actions)
        ]
        self.waiting = True

    def step_wait(self):
        if self._actions is not None:
            from stable_retro import _retro

            selected_actions = [
                env._select_step_action(action)
                for env, action in zip(self.envs, self._actions)
            ]
            masks = np.stack(
                [
                    env.action_to_array(action)[0]
                    for env, action in zip(self.envs, selected_actions)
                ],
            ).astype(np.uint8, copy=False)
            native_emulators = [
                getattr(env.em, "native_emulator", env.em) for env in self.envs
            ]
            datas = [env.data for env in self.envs]
            crops = []
            for env in self.envs:
                width, height = env.em.get_resolution()
                crops.append(env._effective_crop(0, height, width))

            first = self.envs[0]
            obs_list, rewards, dones, infos = _retro.step_repeat_and_process_batch(
                native_emulators,
                datas,
                masks,
                first._frame_skip,
                crops,
                first._obs_resize,
                first._obs_grayscale,
                first._obs_resize_algorithm,
                first._maxpool_last_two,
                self.num_threads,
            )
            observations = []
            out_rewards = []
            out_dones = []
            out_infos = []
            for env, obs, reward, done, info in zip(
                self.envs,
                obs_list,
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool),
                infos,
            ):
                env.img = env._normalize_single_observation(
                    np.asarray(obs, dtype=np.uint8),
                )
                stacked_obs = env._update_frame_stack(env.img)
                clipped_reward = env._clip_reward(float(reward))
                info = dict(info)
                if env.render_mode == "human":
                    env.render()
                if done:
                    terminal_observation = stacked_obs
                    stacked_obs, reset_info = env.reset()
                    info["terminal_observation"] = terminal_observation
                    info["reset_info"] = reset_info
                    info["TimeLimit.truncated"] = False
                observations.append(stacked_obs)
                out_rewards.append(clipped_reward)
                out_dones.append(done)
                out_infos.append(info)
            self._actions = None
            self.waiting = False
            self._observations = np.stack(observations)
            return (
                self._obs(),
                np.asarray(out_rewards, dtype=np.float32),
                np.asarray(out_dones, dtype=bool),
                out_infos,
            )

        results = [future.result() for future in self._futures]
        self._futures = None
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        self._observations = np.stack(observations)
        return (
            self._obs(),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        if self.closed:
            return
        if self.waiting and self._futures is not None:
            for future in self._futures:
                future.result()
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)
        self.closed = True

    def get_images(self):
        return [env.render() for env in self.envs]

    def render(self, mode: str | None = None):
        return [env.render() for env in self.envs]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self.envs[i], attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for i in self._get_indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        results = []
        for i in self._get_indices(indices):
            current = self.envs[i]
            wrapped = False
            while isinstance(current, gym.Wrapper):
                if isinstance(current, wrapper_class):
                    wrapped = True
                    break
                current = current.env
            results.append(wrapped)
        return results


class StableRetroNativeVecEnv(VecEnv):
    """SB3-compatible native vector env for homogeneous stable-retro rollouts.

    This path keeps frame stepping, reward/done evaluation, autoreset,
    preprocessing, and frame stacking inside C++. It currently targets the
    fastest common RL case: single-player image observations without movies or
    screen rotation.
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


class StableRetroSubprocVecEnv(VecEnv):
    """SB3-compatible subprocess vector env with shared-memory observations."""

    def __init__(
        self,
        env_fns,
        start_method: str | None = None,
        copy_observations: bool = False,
    ):
        if VecEnv is object:
            raise ImportError(
                "StableRetroSubprocVecEnv requires stable-baselines3 to be installed",
            )
        self.waiting = False
        self.closed = False
        self.copy_observations = bool(copy_observations)
        n_envs = len(env_fns)
        if n_envs <= 0:
            raise ValueError("env_fns must contain at least one environment factory")
        if start_method is None:
            start_method = "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes,
            self.remotes,
            env_fns,
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        observation_space, action_space = self.remotes[0].recv()
        for remote in self.remotes[1:]:
            other_observation_space, other_action_space = remote.recv()
            if other_observation_space != observation_space:
                raise ValueError("all workers must have the same observation space")
            if other_action_space != action_space:
                raise ValueError("all workers must have the same action space")
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("StableRetroSubprocVecEnv only supports Box observations")

        self.obs_shape = observation_space.shape
        self.obs_dtype = np.dtype(observation_space.dtype)
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod((n_envs, *self.obs_shape))) * self.obs_dtype.itemsize,
        )
        self.observations = np.ndarray(
            (n_envs, *self.obs_shape),
            dtype=self.obs_dtype,
            buffer=self.shm.buf,
        )
        for i, remote in enumerate(self.remotes):
            remote.send(
                (
                    "set_shm",
                    (
                        self.shm.name,
                        self.observations.shape,
                        self.obs_dtype.str,
                        i,
                    ),
                ),
            )
        for remote in self.remotes:
            remote.recv()

        super().__init__(n_envs, observation_space, action_space)

    def _obs(self):
        if self.copy_observations:
            return self.observations.copy()
        return self.observations

    def reset(self):
        for env_idx, remote in enumerate(self.remotes):
            seed = self._seeds[env_idx] if self._seeds else None
            options = self._options[env_idx] if self._options else None
            remote.send(("reset", (seed, options)))
        self.reset_infos = [remote.recv() for remote in self.remotes]
        self._reset_seeds()
        self._reset_options()
        return self._obs()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, dones, infos = zip(*results)
        return (
            self._obs(),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        self.closed = True

    def get_images(self):
        return [self.get_attr("render_mode", i)[0] for i in range(self.num_envs)]

    def render(self, mode: str | None = None):
        for remote in self.remotes:
            remote.send(("render", None))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


class StableRetroChunkedSubprocVecEnv(VecEnv):
    """Shared-memory subprocess vector env with multiple envs per worker."""

    def __init__(
        self,
        env_fns,
        start_method: str | None = None,
        chunk_size: int = 4,
        copy_observations: bool = False,
    ):
        if VecEnv is object:
            raise ImportError(
                "StableRetroChunkedSubprocVecEnv requires stable-baselines3 "
                "to be installed",
            )
        self.waiting = False
        self.closed = False
        self.copy_observations = bool(copy_observations)
        self.chunk_size = int(chunk_size)
        n_envs = len(env_fns)
        if n_envs <= 0:
            raise ValueError("env_fns must contain at least one environment factory")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if start_method is None:
            start_method = "spawn"

        self.chunks = _chunk_env_fns(list(env_fns), self.chunk_size)
        self.worker_indices = []
        start = 0
        for chunk in self.chunks:
            stop = start + len(chunk)
            self.worker_indices.append(list(range(start, stop)))
            start = stop

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(len(self.chunks))],
        )
        self.processes = []
        for work_remote, remote, chunk in zip(
            self.work_remotes,
            self.remotes,
            self.chunks,
        ):
            wrappers = [CloudpickleWrapper(env_fn) for env_fn in chunk]
            args = (work_remote, remote, wrappers)
            process = ctx.Process(target=_chunk_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        spaces = []
        for remote in self.remotes:
            message = remote.recv()
            if message[0] == "error":
                self._join_failed_workers()
                raise RuntimeError(
                    f"worker failed to create chunked environments: {message[1]}",
                )
            spaces.append(message)
        observation_spaces = []
        action_spaces = []
        for _tag, chunk_observation_spaces, chunk_action_spaces in spaces:
            observation_spaces.extend(chunk_observation_spaces)
            action_spaces.extend(chunk_action_spaces)
        observation_space = observation_spaces[0]
        action_space = action_spaces[0]
        for other_observation_space in observation_spaces[1:]:
            if other_observation_space != observation_space:
                raise ValueError("all workers must have the same observation space")
        for other_action_space in action_spaces[1:]:
            if other_action_space != action_space:
                raise ValueError("all workers must have the same action space")
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(
                "StableRetroChunkedSubprocVecEnv only supports Box observations",
            )

        self.obs_shape = observation_space.shape
        self.obs_dtype = np.dtype(observation_space.dtype)
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod((n_envs, *self.obs_shape))) * self.obs_dtype.itemsize,
        )
        self.observations = np.ndarray(
            (n_envs, *self.obs_shape),
            dtype=self.obs_dtype,
            buffer=self.shm.buf,
        )
        for remote, indices in zip(self.remotes, self.worker_indices):
            remote.send(
                (
                    "set_shm",
                    (
                        self.shm.name,
                        self.observations.shape,
                        self.obs_dtype.str,
                        indices,
                    ),
                ),
            )
        for remote in self.remotes:
            remote.recv()

        super().__init__(n_envs, observation_space, action_space)

    def _join_failed_workers(self):
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join()

    def _obs(self):
        if self.copy_observations:
            return self.observations.copy()
        return self.observations

    def reset(self):
        for remote, indices in zip(self.remotes, self.worker_indices):
            payload = []
            for env_idx in indices:
                seed = self._seeds[env_idx] if self._seeds else None
                options = self._options[env_idx] if self._options else None
                payload.append((seed, options))
            remote.send(("reset", payload))
        self.reset_infos = [info for remote in self.remotes for info in remote.recv()]
        self._reset_seeds()
        self._reset_options()
        return self._obs()

    def step_async(self, actions):
        for remote, indices in zip(self.remotes, self.worker_indices):
            remote.send(("step", np.asarray(actions)[indices]))
        self.waiting = True

    def step_wait(self):
        chunk_results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards = []
        dones = []
        infos = []
        for chunk_rewards, chunk_dones, chunk_infos in chunk_results:
            rewards.extend(chunk_rewards)
            dones.extend(chunk_dones)
            infos.extend(chunk_infos)
        return (
            self._obs(),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        self.closed = True

    def get_images(self):
        return [self.get_attr("render_mode", i)[0] for i in range(self.num_envs)]

    def render(self, mode: str | None = None):
        images = []
        for remote in self.remotes:
            remote.send(("render", None))
        for remote in self.remotes:
            images.extend(remote.recv())
        return images

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        groups = self._group_indices(indices)
        results = [None] * sum(
            len(result_positions) for *_rest, result_positions in groups
        )
        for remote, local_indices, _result_positions in groups:
            remote.send(("get_attr", (attr_name, local_indices)))
        for remote, _local_indices, result_positions in groups:
            for result_position, value in zip(result_positions, remote.recv()):
                results[result_position] = value
        return results

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        groups = self._group_indices(indices)
        for remote, local_indices, _result_positions in groups:
            remote.send(("set_attr", (attr_name, value, local_indices)))
        for remote, _local_indices, _result_positions in groups:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        groups = self._group_indices(indices)
        results = [None] * sum(
            len(result_positions) for *_rest, result_positions in groups
        )
        for remote, local_indices, _result_positions in groups:
            remote.send(
                (
                    "env_method",
                    (method_name, method_args, method_kwargs, local_indices),
                ),
            )
        for remote, _local_indices, result_positions in groups:
            for result_position, value in zip(result_positions, remote.recv()):
                results[result_position] = value
        return results

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        groups = self._group_indices(indices)
        results = [None] * sum(
            len(result_positions) for *_rest, result_positions in groups
        )
        for remote, local_indices, _result_positions in groups:
            remote.send(("is_wrapped", (wrapper_class, local_indices)))
        for remote, _local_indices, result_positions in groups:
            for result_position, value in zip(result_positions, remote.recv()):
                results[result_position] = value
        return results

    def _group_indices(self, indices):
        indices = list(self._get_indices(indices))
        groups = []
        for remote, worker_indices in zip(self.remotes, self.worker_indices):
            local_indices = []
            result_positions = []
            for result_position, index in enumerate(indices):
                if index in worker_indices:
                    local_indices.append(worker_indices.index(index))
                    result_positions.append(result_position)
            if local_indices:
                groups.append((remote, local_indices, result_positions))
        return groups


__all__ = [
    "StableRetroNativeVecEnv",
    "StableRetroThreadedVecEnv",
    "StableRetroSubprocVecEnv",
    "StableRetroChunkedSubprocVecEnv",
]

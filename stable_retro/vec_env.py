"""Shared-memory vector environments for stable-retro workers."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import resource_tracker
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
                name, shape, dtype, obs_index, unregister_resource = data
                shm = shared_memory.SharedMemory(name=name)
                if unregister_resource:
                    resource_tracker.unregister(shm._name, "shared_memory")
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
                        start_method == "fork",
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
        self.shm.unlink()
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


__all__ = ["StableRetroSubprocVecEnv"]

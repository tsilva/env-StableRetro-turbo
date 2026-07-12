"""Atari-specialized vector environment exposed through stable_retro."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.vector import VectorEnv
import stable_retro.data as retro_data


_STABLE_RETRO_TO_ALE_GAME = {
    "Breakout-Atari2600-v0": "breakout",
    "MsPacman-Atari2600-v0": "ms_pacman",
}


def _atari_vector_env_type():
    try:
        from ale_py.vector_env import AtariVectorEnv
    except ImportError as exc:
        raise ImportError(
            "stable_retro.AtariVecEnv requires ale-py-turbo with native vector support"
        ) from exc
    return AtariVectorEnv


class AtariVecEnv(VectorEnv):
    """Stable Retro facade for ale-py-turbo's native Atari vector backend.

    Stable Retro Atari IDs are translated to ALE ROM names while the native
    backend retains ownership of stepping, preprocessing, and masked reset.
    """

    def __init__(self, game: str, **kwargs: Any):
        native_game = _STABLE_RETRO_TO_ALE_GAME.get(str(game), str(game))
        self.game = str(game)
        self.native_game = native_game
        if self.game in _STABLE_RETRO_TO_ALE_GAME and "rom_path" not in kwargs:
            kwargs["rom_path"] = retro_data.get_original_romfile_path(
                self.game,
                retro_data.Integrations.STABLE,
            )
        self.env = _atari_vector_env_type()(native_game, **kwargs)
        self._observations: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._observations = np.asarray(observations)
        return observations, infos

    def step(self, actions):
        result = self.env.step(actions)
        self._observations = np.asarray(result[0])
        return result

    def get_images(self):
        if self._observations is None:
            return []
        observations = self._observations
        if observations.ndim not in (4, 5):
            raise ValueError(
                f"unsupported ALE observation shape for rendering: {observations.shape}"
            )
        frames = observations[:, -1]
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        elif frames.ndim == 4 and frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        return [np.asarray(frame) for frame in frames]

    def close(self):
        env = self.__dict__.get("env")
        if env is not None:
            return env.close()
        return None


__all__ = ["AtariVecEnv"]

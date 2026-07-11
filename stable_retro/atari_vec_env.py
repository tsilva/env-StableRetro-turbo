"""The shared-core Atari vector environment."""

from __future__ import annotations

from gymnasium.vector import AutoresetMode

from stable_retro.enums import Actions, State

try:
    from ale_py import AtariVectorEnv as _AleAtariVectorEnv
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the extra
    raise ModuleNotFoundError(
        "AtariVecEnv requires ale-py; install stable-retro-turbo[atari]",
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


class AtariVecEnv(_AleAtariVectorEnv):
    """High-throughput Atari vector environment.

    This is the sole supported Atari backend. It shares reentrant emulator code
    across lanes and uses native discrete Atari actions. Stable Retro ``.state``
    files belong to the removed libretro backend and are unsupported.
    """

    backend = "atari-v1"

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
                "use state=State.NONE",
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

        self.stable_retro_game = str(game)
        self.ale_game = ale_game_id(game)
        super().__init__(
            self.ale_game,
            num_envs,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
            max_num_frames_per_episode=max_episode_steps,
            repeat_action_probability=float(sticky_action_prob),
            full_action_space=full_action_space,
            autoreset_mode=autoreset_mode,
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


__all__ = ["AtariVecEnv", "ale_game_id"]

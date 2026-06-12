import gc
import gzip
import json
import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.ezpickle import EzPickle

import stable_retro as retro
import stable_retro.data

__all__ = ["RetroEnv"]


class RetroEnv(gym.Env, EzPickle):
    """
    Gym Retro environment class

    Provides a Gym interface to classic video games
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 60.0}

    def __init__(
        self,
        game,
        state=retro.State.DEFAULT,
        scenario=None,
        info=None,
        use_restricted_actions=retro.Actions.FILTERED,
        record=False,
        players=1,
        inttype=retro.data.Integrations.STABLE,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",
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
    ):
        """Initialize a Retro environment for a specific game/state configuration."""
        if inttype is retro.data.Integrations.DEFAULT or isinstance(
            inttype,
            retro.data.DefaultIntegrations,
        ):
            retro.data.DefaultIntegrations._init()
            inttype = retro.data.DefaultIntegrations.DEFAULT

        EzPickle.__init__(
            self,
            game,
            state,
            scenario,
            info,
            use_restricted_actions,
            record,
            players,
            inttype,
            obs_type,
            render_mode,
            obs_resize,
            obs_crop,
            obs_grayscale,
            obs_resize_algorithm,
            frame_skip,
            frame_stack,
            maxpool_last_two,
            noop_reset_max,
            sticky_action_prob,
            reward_clip,
        )
        if not hasattr(self, "spec"):
            self.spec = None
        self._obs_type = obs_type
        self._obs_resize = self._normalize_obs_resize(obs_resize)
        self._obs_crop = self._normalize_obs_crop(obs_crop)
        self._obs_grayscale = bool(obs_grayscale)
        self._obs_resize_algorithm = self._normalize_obs_resize_algorithm(
            obs_resize_algorithm,
        )
        self._obs_resize_cache = {}
        self._frame_skip = self._normalize_positive_int(frame_skip, "frame_skip")
        self._frame_stack = self._normalize_positive_int(frame_stack, "frame_stack")
        self._maxpool_last_two = bool(maxpool_last_two)
        self._noop_reset_max = self._normalize_nonnegative_int(
            noop_reset_max,
            "noop_reset_max",
        )
        self._sticky_action_prob = float(sticky_action_prob)
        if not 0.0 <= self._sticky_action_prob <= 1.0:
            raise ValueError("sticky_action_prob must be between 0.0 and 1.0")
        self._reward_clip = reward_clip
        self._frame_stack_buffer = []
        self._last_action = None
        self.img = None
        self.ram = None
        self.viewer = None
        self.gamename = game
        self.statename = state
        self.initial_state = None
        self.players = players

        # Don't return multiple rewards in multiplayer mode by default
        # as stable-baselines3 vectorized environments doesn't support it
        self.multi_rewards = False

        metadata = {}
        rom_path = retro.data.get_original_romfile_path(game, inttype)
        metadata_path = retro.data.get_file_path(game, "metadata.json", inttype)

        if state == retro.State.NONE:
            self.statename = None
        elif state == retro.State.DEFAULT:
            self.statename = None
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if "default_player_state" in metadata and self.players <= len(
                    metadata["default_player_state"],
                ):
                    self.statename = metadata["default_player_state"][self.players - 1]
                elif "default_state" in metadata:
                    self.statename = metadata["default_state"]
                else:
                    self.statename = None
            except (OSError, json.JSONDecodeError):
                pass

        if self.statename:
            self.load_state(self.statename, inttype)

        self.data = retro.data.GameData()

        if info is None:
            info = "data"

        if info.endswith(".json"):
            # assume it's a path
            info_path = info
        else:
            info_path = retro.data.get_file_path(game, info + ".json", inttype)

        if scenario is None:
            scenario = "scenario"

        if scenario.endswith(".json"):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = retro.data.get_file_path(game, scenario + ".json", inttype)

        self.system = retro.get_romfile_system(rom_path)

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        self.em = retro.RetroEmulator(rom_path)
        self.em.configure_data(self.data)
        self.em.step()

        core = retro.get_system_info(self.system)
        self.buttons = core["buttons"]
        self.num_buttons = len(self.buttons)

        try:
            assert self.data.load(
                info_path,
                scenario_path,
            ), "Failed to load info ({}) or scenario ({})".format(
                info_path,
                scenario_path,
            )
        except Exception:
            del self.em
            raise

        self.button_combos = self.data.valid_actions()
        if use_restricted_actions == retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos**players)
        elif use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete(
                [len(combos) for combos in self.button_combos] * players,
            )
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)

        if self._obs_type == retro.Observations.RAM:
            shape = self.get_ram().shape
        else:
            img = [self.get_screen(p, apply_rotation=True) for p in range(players)]
            shape = img[0].shape
        shape = self._stacked_obs_shape(shape)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

        self.use_restricted_actions = use_restricted_actions
        self.movie = None
        self.movie_id = 0
        self.movie_path = None
        if record is True:
            self.auto_record()
        elif record is not False:
            self.auto_record(record)

        self.render_mode = render_mode

    def _update_obs(self):
        if self._obs_type == retro.Observations.RAM:
            self.ram = self.get_ram()
            return self._update_frame_stack(self.ram)
        elif self._obs_type == retro.Observations.IMAGE:
            self.img = self.get_screen(apply_rotation=True)
            return self._update_frame_stack(self.img)
        else:
            raise ValueError(f"Unrecognized observation type: {self._obs_type}")

    def _rotation_steps(self):
        try:
            rotation = int(self.em.get_rotation())
        except AttributeError:
            return 0
        return rotation % 4

    def _apply_rotation(self, image):
        steps = self._rotation_steps()
        if steps == 0:
            return image
        if steps == 1:
            return np.flipud(np.swapaxes(image, 0, 1))
        if steps == 2:
            return np.flipud(np.fliplr(image))
        if steps == 3:
            return np.fliplr(np.swapaxes(image, 0, 1))
        return image

    @staticmethod
    def _normalize_obs_resize(obs_resize):
        if obs_resize is None:
            return None
        if len(obs_resize) != 2:
            raise ValueError("obs_resize must be a (height, width) pair")
        height, width = (int(obs_resize[0]), int(obs_resize[1]))
        if height <= 0 or width <= 0:
            raise ValueError("obs_resize height and width must be positive")
        return height, width

    @staticmethod
    def _normalize_obs_crop(obs_crop):
        if obs_crop is None:
            return None
        if len(obs_crop) != 4:
            raise ValueError("obs_crop must be a (top, bottom, left, right) tuple")
        top, bottom, left, right = (int(v) for v in obs_crop)
        if min(top, bottom, left, right) < 0:
            raise ValueError("obs_crop values must be non-negative")
        return top, bottom, left, right

    @staticmethod
    def _normalize_obs_resize_algorithm(obs_resize_algorithm):
        algorithm = str(obs_resize_algorithm).lower()
        aliases = {
            "linear": "bilinear",
            "box": "area",
        }
        algorithm = aliases.get(algorithm, algorithm)
        if algorithm not in {"nearest", "bilinear", "area"}:
            raise ValueError(
                "obs_resize_algorithm must be one of: nearest, bilinear, area",
            )
        return algorithm

    @staticmethod
    def _normalize_positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _normalize_nonnegative_int(value, name):
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _stacked_obs_shape(self, shape):
        if self._frame_stack == 1:
            return shape
        if len(shape) == 1:
            return (shape[0] * self._frame_stack,)
        return (*shape[:-1], shape[-1] * self._frame_stack)

    def _stack_frames(self):
        if len(self._frame_stack_buffer) != self._frame_stack:
            raise RuntimeError("frame stack buffer is not initialized")
        if self._frame_stack == 1:
            return self._frame_stack_buffer[-1]
        if self._frame_stack_buffer[-1].ndim == 1:
            return np.concatenate(self._frame_stack_buffer, axis=0)
        return np.concatenate(self._frame_stack_buffer, axis=-1)

    def _update_frame_stack(self, obs):
        if self._frame_stack == 1:
            return obs
        obs = np.asarray(obs, dtype=np.uint8)
        if not self._frame_stack_buffer:
            self._frame_stack_buffer = [obs.copy() for _ in range(self._frame_stack)]
        else:
            self._frame_stack_buffer.append(obs.copy())
            del self._frame_stack_buffer[: -self._frame_stack]
        return self._stack_frames()

    def _reset_frame_stack(self, obs):
        obs = np.asarray(obs, dtype=np.uint8)
        if self._frame_stack == 1:
            return obs
        self._frame_stack_buffer = [obs.copy() for _ in range(self._frame_stack)]
        return self._stack_frames()

    def _clip_reward(self, reward):
        if not self._reward_clip:
            return reward
        if self._reward_clip is True:
            low, high = -1.0, 1.0
        else:
            low, high = self._reward_clip
        if isinstance(reward, (list, tuple, np.ndarray)):
            return np.clip(reward, low, high).tolist()
        return float(np.clip(reward, low, high))

    @staticmethod
    def _add_reward(left, right):
        if left is None:
            return right
        if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
            right,
            (list, tuple, np.ndarray),
        ):
            return (
                np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
            ).tolist()
        return left + right

    def _apply_obs_crop(self, image):
        if self._obs_crop is None:
            return image
        top, bottom, left, right = self._obs_crop
        height, width = image.shape[:2]
        y2 = height - bottom if bottom else height
        x2 = width - right if right else width
        if top >= y2 or left >= x2:
            raise ValueError("obs_crop removes the entire observation")
        return image[top:y2, left:x2]

    def _effective_crop(self, player, raw_height, raw_width):
        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > raw_width:
            right_edge = raw_width
        else:
            right_edge = x + w
        if not h or y + h > raw_height:
            bottom_edge = raw_height
        else:
            bottom_edge = y + h
        top = y
        left = x
        if self._obs_crop is not None:
            obs_top, obs_bottom, obs_left, obs_right = self._obs_crop
            top += obs_top
            left += obs_left
            bottom_edge -= obs_bottom
            right_edge -= obs_right
        if top >= bottom_edge or left >= right_edge:
            raise ValueError("obs_crop removes the entire observation")
        return (
            int(top),
            int(raw_height - bottom_edge),
            int(left),
            int(raw_width - right_edge),
        )

    def _native_processed_screen(self, player):
        if player != 0 or self._rotation_steps() != 0:
            return None
        if os.environ.get("STABLE_RETRO_DISABLE_NATIVE_IMAGEOPS"):
            return None
        if not hasattr(self.em, "get_processed_screen"):
            return None
        width, height = self.em.get_resolution()
        crop = self._effective_crop(player, height, width)
        return self.em.get_processed_screen(
            crop,
            self._obs_resize,
            self._obs_grayscale,
            self._obs_resize_algorithm,
        )

    def _apply_obs_grayscale(self, image):
        if not self._obs_grayscale:
            return image
        if image.ndim == 2:
            return image[:, :, None]
        gray = (
            image[:, :, 0].astype(np.uint16) * 77
            + image[:, :, 1].astype(np.uint16) * 150
            + image[:, :, 2].astype(np.uint16) * 29
            + 128
        ) >> 8
        return gray.astype(np.uint8)[:, :, None]

    def _resize_obs(self, image):
        if self._obs_resize is None:
            return image
        height, width = self._obs_resize
        src_height, src_width = image.shape[:2]
        key = (src_height, src_width, height, width, self._obs_resize_algorithm)
        indices = self._obs_resize_cache.get(key)
        if indices is None:
            if self._obs_resize_algorithm == "nearest":
                y_idx = np.linspace(0, src_height - 1, height).astype(np.intp)
                x_idx = np.linspace(0, src_width - 1, width).astype(np.intp)
                indices = (y_idx, x_idx)
            elif self._obs_resize_algorithm == "bilinear":
                y = np.linspace(0, src_height - 1, height, dtype=np.float32)
                x = np.linspace(0, src_width - 1, width, dtype=np.float32)
                y0 = np.floor(y).astype(np.intp)
                x0 = np.floor(x).astype(np.intp)
                y1 = np.minimum(y0 + 1, src_height - 1)
                x1 = np.minimum(x0 + 1, src_width - 1)
                wy = (y - y0).astype(np.float32)[:, None, None]
                wx = (x - x0).astype(np.float32)[None, :, None]
                indices = (y0, y1, x0, x1, wy, wx)
            else:
                if height > src_height or width > src_width:
                    raise ValueError("area resize only supports downscaling")
                y_edges = np.linspace(0, src_height, height + 1).astype(np.intp)
                x_edges = np.linspace(0, src_width, width + 1).astype(np.intp)
                y_edges[1:] = np.maximum(y_edges[1:], y_edges[:-1] + 1)
                x_edges[1:] = np.maximum(x_edges[1:], x_edges[:-1] + 1)
                y_edges[-1] = src_height
                x_edges[-1] = src_width
                indices = (y_edges, x_edges)
            self._obs_resize_cache[key] = indices
        if self._obs_resize_algorithm == "nearest":
            y_idx, x_idx = indices
            return image[y_idx][:, x_idx]
        if self._obs_resize_algorithm == "bilinear":
            y0, y1, x0, x1, wy, wx = indices
            top = (
                image[y0][:, x0].astype(np.float32) * (1.0 - wx)
                + image[y0][:, x1].astype(np.float32) * wx
            )
            bottom = (
                image[y1][:, x0].astype(np.float32) * (1.0 - wx)
                + image[y1][:, x1].astype(np.float32) * wx
            )
            return np.clip(top * (1.0 - wy) + bottom * wy, 0, 255).astype(np.uint8)
        y_edges, x_edges = indices
        integral = image.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)
        integral = np.pad(integral, ((1, 0), (1, 0), (0, 0)), mode="constant")
        y0 = y_edges[:-1]
        y1 = y_edges[1:]
        x0 = x_edges[:-1]
        x1 = x_edges[1:]
        sums = (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )
        pixels = ((y1 - y0)[:, None] * (x1 - x0)[None, :])[:, :, None]
        return (sums // pixels).astype(np.uint8)

    def action_to_array(self, a):
        """Convert an action-space value into per-player button-mask arrays."""
        actions = []
        for p in range(self.players):
            action = 0
            if self.use_restricted_actions == retro.Actions.DISCRETE:
                for combo in self.button_combos:
                    current = a % len(combo)
                    a //= len(combo)
                    action |= combo[current]
            elif self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
                ap = a[self.num_buttons * p : self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    buttons = self.button_combos[i]
                    action |= buttons[ap[i]]
            else:
                ap = a[self.num_buttons * p : self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    action |= int(ap[i]) << i
                if self.use_restricted_actions == retro.Actions.FILTERED:
                    action = self.data.filter_action(action)
            ap = np.zeros([self.num_buttons], np.uint8)
            for i in range(self.num_buttons):
                ap[i] = (action >> i) & 1
            actions.append(ap)
        return actions

    def _noop_action(self):
        if self.use_restricted_actions == retro.Actions.DISCRETE:
            return 0
        if self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            return np.zeros([len(self.button_combos) * self.players], dtype=np.int64)
        return np.zeros([self.num_buttons * self.players], dtype=np.uint8)

    def _select_step_action(self, a):
        if (
            self._last_action is not None
            and self._sticky_action_prob > 0.0
            and self.np_random.random() < self._sticky_action_prob
        ):
            return self._last_action
        self._last_action = np.array(a, copy=True) if isinstance(a, np.ndarray) else a
        return a

    def _set_action(self, a):
        for p, ap in enumerate(self.action_to_array(a)):
            if self.movie:
                for i in range(self.num_buttons):
                    self.movie.set_key(i, ap[i], p)
            self.em.set_button_mask(ap, p)

    def _advance_one_frame(self):
        if self.movie:
            self.movie.step()
        self.em.step()
        self.data.update_ram()
        return self.compute_step()

    def step(self, a):
        """Advance one emulator frame and return Gymnasium step outputs."""
        if self.img is None and self.ram is None:
            raise RuntimeError("Please call env.reset() before env.step()")

        action = self._select_step_action(a)
        self._set_action(action)

        total_rew = None
        done = False
        info = {}
        recent_obs = []
        for _ in range(self._frame_skip):
            rew, done, info = self._advance_one_frame()
            total_rew = self._add_reward(total_rew, rew)
            if self._maxpool_last_two and self._obs_type == retro.Observations.IMAGE:
                recent_obs.append(self.get_screen(apply_rotation=True))
                del recent_obs[:-2]
            if done:
                break

        if (
            self._maxpool_last_two
            and self._obs_type == retro.Observations.IMAGE
            and len(recent_obs) == 2
        ):
            self.img = np.maximum(recent_obs[0], recent_obs[1])
            ob = self._update_frame_stack(self.img)
        else:
            ob = self._update_obs()
        rew = self._clip_reward(total_rew if total_rew is not None else 0.0)

        if self.render_mode == "human":
            self.render()

        return ob, rew, bool(done), False, dict(info)

    def reset(self, seed=None, options=None):
        """Reset emulator state and return the initial observation and info."""
        super().reset(seed=seed)

        if self.initial_state:
            self.em.set_state(self.initial_state)
        self._last_action = None
        self._frame_stack_buffer = []
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
            self.record_movie(
                os.path.join(
                    self.movie_path,
                    "%s-%s-%06d.bk2" % (self.gamename, rel_statename, self.movie_id),
                ),
            )
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.data.update_ram()

        if self._noop_reset_max:
            noop_action = self._noop_action()
            noop_count = int(self.np_random.integers(0, self._noop_reset_max + 1))
            self._set_action(noop_action)
            for _ in range(noop_count):
                _rew, done, _info = self._advance_one_frame()
                if done:
                    break

        if self.render_mode == "human":
            self.render()

        if self._obs_type == retro.Observations.RAM:
            self.ram = self.get_ram()
            ob = self.ram
        else:
            self.img = self.get_screen(apply_rotation=True)
            ob = self.img
        return self._reset_frame_stack(ob), {}

    def render(self):
        """Render the current frame in human mode or return an RGB array."""
        mode = self.render_mode

        img = self.img
        if img is None:
            img = self.get_screen(apply_rotation=True)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            if self.viewer is None:
                from stable_retro.rendering import SimpleImageViewer

                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img, rotation=0)
            return self.viewer.isopen

    def close(self):
        """Release emulator and viewer resources."""
        if hasattr(self, "em"):
            del self.em
        if self.viewer:
            self.viewer.close()

    def get_action_meaning(self, act):
        """Return human-readable button names for an encoded action."""
        actions = []
        for p, action in enumerate(self.action_to_array(act)):
            actions.append(
                [self.buttons[i] for i in np.extract(action, np.arange(len(action)))],
            )
        if self.players == 1:
            return actions[0]
        return actions

    def set_value(self, name, val):
        self.data.set_value(name, val)

    def get_ram(self):
        """Return concatenated emulator RAM blocks as a uint8 array."""
        blocks = []
        for offset in sorted(self.data.memory.blocks):
            arr = np.frombuffer(self.data.memory.blocks[offset], dtype=np.uint8)
            blocks.append(arr)
        return np.concatenate(blocks)

    def get_screen(self, player=0, apply_rotation=False):
        """Return the current screen, optionally cropped and rotation-corrected."""
        if apply_rotation:
            native = self._native_processed_screen(player)
            if native is not None:
                return native
        img = self.em.get_screen()
        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > img.shape[1]:
            w = img.shape[1]
        else:
            w += x
        if not h or y + h > img.shape[0]:
            h = img.shape[0]
        else:
            h += y
        if x == 0 and y == 0 and w == img.shape[1] and h == img.shape[0]:
            result = img
        else:
            result = img[y:h, x:w]
        if apply_rotation:
            result = self._apply_rotation(result)
        result = self._apply_obs_crop(result)
        result = self._apply_obs_grayscale(result)
        result = self._resize_obs(result)
        return result

    def load_state(self, statename, inttype=retro.data.Integrations.DEFAULT):
        """Load a named save-state file into ``self.initial_state``."""
        if not statename.endswith(".state"):
            statename += ".state"

        with gzip.open(
            retro.data.get_file_path(self.gamename, statename, inttype),
            "rb",
        ) as fh:
            self.initial_state = fh.read()

        self.statename = statename

    def compute_step(self):
        """Compute reward, done flag, and info dictionary from current RAM data."""
        if self.players > 1 and self.multi_rewards:
            reward = [self.data.current_reward(p) for p in range(self.players)]
        else:
            reward = self.data.current_reward()
        done = self.data.is_done()
        return reward, done, self.data.lookup_all()

    def record_movie(self, path):
        """Start recording gameplay input to a BK2 movie at ``path``."""
        self.movie = retro.Movie(path, True, self.players)
        self.movie.configure(
            self.gamename,
            getattr(self.em, "native_emulator", self.em),
        )
        if self.initial_state:
            self.movie.set_state(self.initial_state)

    def stop_record(self):
        """Stop movie recording and clear recording state."""
        self.movie_path = None
        self.movie_id = 0
        if self.movie:
            self.movie.close()
            self.movie = None

    def auto_record(self, path=None):
        """Enable automatic per-episode movie recording to a directory."""
        if not path:
            path = os.getcwd()
        self.movie_path = path

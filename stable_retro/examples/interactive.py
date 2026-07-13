"""
Interact with Gym environments using the keyboard

An adapter object is defined for each environment to map keyboard commands to actions and extract observations as pixels.
"""

import abc
import argparse
import ctypes
import math
import time

import numpy as np
import pyglet

# Importing the CLI must not require an active display. The real OpenGL context
# is created when Interactive opens its visible window.
pyglet.options["shadow_window"] = False
from pyglet import gl
from pyglet.window import key as keycodes

import stable_retro as retro


def _environment_aspect_ratio(env, image_width, image_height):
    emulator = getattr(env, "em", None)
    get_aspect_ratio = getattr(emulator, "get_aspect_ratio", None)
    if get_aspect_ratio is not None:
        aspect_ratio = float(get_aspect_ratio())
        if math.isfinite(aspect_ratio) and aspect_ratio > 0:
            return aspect_ratio
    return image_width / image_height


def _window_dimensions(
    image_width,
    image_height,
    aspect_ratio,
    screen_width,
    screen_height,
):
    """Choose the largest pixel-aligned window that fits on the display."""
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        aspect_ratio = image_width / image_height

    base_width = float(image_width)
    base_height = base_width / aspect_ratio
    max_width = screen_width * 0.9
    max_height = screen_height * 0.9
    scale = min(max_width / base_width, max_height / base_height)
    if scale >= 1:
        scale = math.floor(scale)

    width = max(1, round(base_width * scale))
    height = max(1, round(width / aspect_ratio))
    return width, height


def _observation_to_rgb(observation):
    """Turn an HWC image observation into an RGB image without altering pixels.

    Frame-stacked grayscale observations are tiled left-to-right so every frame
    given to the policy remains visible.
    """
    observation = np.asarray(observation, dtype=np.uint8)
    if observation.ndim != 3:
        raise ValueError("observation viewer requires an HWC image observation")
    if observation.shape[2] == 3:
        return observation
    if observation.shape[2] == 1:
        return np.repeat(observation, 3, axis=2)

    frames = [
        np.repeat(observation[..., channel : channel + 1], 3, axis=2)
        for channel in range(observation.shape[2])
    ]
    return np.concatenate(frames, axis=1)


class _ImageWindow:
    """A small nearest-neighbour pyglet image window."""

    def __init__(self, image, title, on_close):
        image_height, image_width = image.shape[:2]
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        width, height = _window_dimensions(
            image_width,
            image_height,
            image_width / image_height,
            screen.width / 2,
            screen.height * 0.9,
        )
        self.window = pyglet.window.Window(width=width, height=height, caption=title)
        self.window.on_close = on_close
        self.image = image

        gl.glEnable(gl.GL_TEXTURE_2D)
        self.texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self.texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

    def set_image(self, image):
        self.image = image

    def draw(self):
        self.window.switch_to()
        self.window.dispatch_events()
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        video_buffer = ctypes.cast(self.image.tobytes(), ctypes.POINTER(ctypes.c_short))
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.image.shape[1],
            self.image.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )
        w, h = self.window.width, self.window.height
        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [0, 0, w, 0, w, h, 0, h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )
        self.window.flip()


class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """

    def __init__(
        self,
        env,
        sync=True,
        tps=60,
        aspect_ratio=None,
        show_obs=False,
        reset_action=None,
    ):
        self._env = env
        self._reset_action = reset_action
        obs, _info = self._reset_environment()
        self._observation = obs
        self._image = self.get_image(obs, env)
        assert (
            len(self._image.shape) == 3 and self._image.shape[2] == 3
        ), "must be an RGB image"
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = _environment_aspect_ratio(
                env,
                image_width,
                image_height,
            )

        # Pick a large pixel-aligned size without distorting the core's display ratio.
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        win_width, win_height = _window_dimensions(
            image_width,
            image_height,
            aspect_ratio,
            screen.width,
            screen.height,
        )

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        self._win = win
        self._closed = False

        # self._render_human = render_human
        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4
        self._observation_window = None
        if show_obs:
            self._observation_window = _ImageWindow(
                self.get_observation_image(obs, reset=True),
                "Preprocessed observation",
                self._on_close,
            )

    def _reset_environment(self):
        """Reset the environment and apply any launcher-specific start action."""
        obs, info = self._env.reset()
        if self._reset_action is not None:
            obs, _reward, _terminated, _truncated, _step_info = self._env.step(
                self._reset_action,
            )
        return obs, info

    def get_observation_image(self, obs, reset=False):
        """Return the RGB visualization for an environment observation."""
        return _observation_to_rgb(obs)

    def _update(self, dt):
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._win.close()
                return

            if self._closed:
                return

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, terminated, truncated, _info = self._env.step(act)
                done = terminated or truncated
                self._observation = obs
                self._image = self.get_image(obs, self._env)
                if self._observation_window is not None:
                    observation_image = self.get_observation_image(obs)
                    if observation_image is not None:
                        self._observation_window.set_image(observation_image)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)
                if self._sync:
                    done_int = int(done)  # shorter than printing True/False
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}"
                    print(mess)
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = (
                        self._episode_returns - self._prev_episode_returns
                    )
                    self._prev_episode_returns = self._episode_returns
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}"
                    print(mess)

                if done:
                    obs, _info = self._reset_environment()
                    self._observation = obs
                    self._image = self.get_image(obs, self._env)
                    if self._observation_window is not None:
                        self._observation_window.set_image(
                            self.get_observation_image(obs, reset=True),
                        )
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(
            self._image.tobytes(),
            ctypes.POINTER(ctypes.c_short),
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._image.shape[1],
            self._image.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        """Release the emulator once and let the outer loop exit cleanly.

        Pyglet dispatches close handlers inside its event loop.  Raising
        ``SystemExit`` there can be caught while the manual render loop carries
        on, leaving a closed ``RetroEnv`` to receive one more ``step`` call.
        """
        if self._closed:
            return
        self._closed = True
        self._env.close()

    @abc.abstractmethod
    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an rgb array to display to the user
        """
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment

        For sync environments, keys is a list of keys that have been pressed since the last step
        For async environments, keys is a list of keys currently held down
        """
        pass

    def run(self):
        """
        Run the interactive window until the user quits
        """
        # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
        # and also involves inverting your code to run inside the pyglet framework
        # avoid both by using a while loop
        prev_frame_time = time.time()
        while not self._closed:
            self._win.switch_to()
            self._win.dispatch_events()
            if self._closed:
                break
            now = time.time()
            self._update(now - prev_frame_time)
            if self._closed:
                break
            prev_frame_time = now
            self._draw()
            self._win.flip()
            if self._observation_window is not None and not self._closed:
                self._observation_window.draw()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games
    """

    def __init__(self, game, state, scenario, record, show_obs=False):
        self._observation_sample_interval = 4
        self._observation_frame_stack = 4
        self._observation_sample_steps = 0
        self._observation_frames = []
        env_kwargs = {}
        if show_obs:
            env_kwargs = {
                "obs_resize": (84, 84),
                "obs_crop": (32, 0, 0, 0),
                "obs_grayscale": True,
                "obs_resize_algorithm": "area",
            }
        env = retro.make(
            game=game,
            state=state,
            scenario=scenario,
            record=record,
            render_mode="rgb_array",
            **env_kwargs,
        )
        self._buttons = env.buttons
        super().__init__(
            env=env,
            sync=False,
            tps=60,
            show_obs=show_obs,
            reset_action=None,
        )

    def get_image(self, _obs, env):
        return env.render()

    def get_observation_image(self, obs, reset=False):
        """Sample a four-frame policy observation without slowing RGB display."""
        frame = np.asarray(obs, dtype=np.uint8)
        if reset:
            self._observation_sample_steps = 0
            self._observation_frames = [
                frame.copy() for _ in range(self._observation_frame_stack)
            ]
        else:
            self._observation_sample_steps += 1
            if self._observation_sample_steps % self._observation_sample_interval:
                return None
            self._observation_frames.append(frame.copy())
            del self._observation_frames[: -self._observation_frame_stack]
        stacked = np.concatenate(self._observation_frames, axis=2)
        return _observation_to_rgb(stacked)

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            "BUTTON": "Z" in keys,
            "A": "Z" in keys,
            "B": "X" in keys,
            "C": "C" in keys,
            "X": "A" in keys,
            "Y": "S" in keys,
            "Z": "D" in keys,
            "L": "Q" in keys,
            "R": "W" in keys,
            "UP": "UP" in keys,
            "DOWN": "DOWN" in keys,
            "LEFT": "LEFT" in keys,
            "RIGHT": "RIGHT" in keys,
            "MODE": "TAB" in keys,
            "SELECT": "TAB" in keys,
            "PAUSE": "ENTER" in keys,
            "RESET": "ENTER" in keys,
            "START": "ENTER" in keys,
        }
        return [inputs[b] for b in self._buttons]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--record", default=None, nargs="?", const=True)
    args = parser.parse_args()

    ia = RetroInteractive(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
    )
    ia.run()


if __name__ == "__main__":
    main()

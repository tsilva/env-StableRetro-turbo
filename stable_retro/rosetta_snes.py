import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from stable_retro._retro import RetroEmulator as NativeRetroEmulator

SNES_EXTENSIONS = {".sfc", ".smc"}
HELPER_NAME = "rosetta_snes_helper"
DISABLE_ENV = "STABLE_RETRO_DISABLE_ROSETTA_SNES"
FORCE_ENV = "STABLE_RETRO_FORCE_ROSETTA_SNES"


def should_use_rosetta_snes(rom_path):
    if os.environ.get(DISABLE_ENV) == "1":
        return False
    if os.environ.get(FORCE_ENV) == "1":
        return True
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and Path(rom_path).suffix.lower() in SNES_EXTENSIONS
    )


class RosettaSnesError(RuntimeError):
    pass


class _RosettaSnesHelper:
    def __init__(self, rom_path):
        package_dir = Path(__file__).resolve().parent
        helper_path = package_dir / "helpers" / HELPER_NAME
        core_dir = package_dir / "cores_rosetta"

        missing = [
            str(path)
            for path in (helper_path, core_dir / "snes9x_libretro.dylib")
            if not path.exists()
        ]
        if missing:
            raise RosettaSnesError(
                "Rosetta SNES assets are missing from this install: "
                + ", ".join(missing),
            )

        try:
            probe = subprocess.run(
                ["arch", "-x86_64", "/usr/bin/true"],
                capture_output=True,
                check=True,
            )
            del probe
        except Exception as exc:
            raise RosettaSnesError(
                "Rosetta is required for SNES on Apple Silicon. Install it with "
                "`softwareupdate --install-rosetta --agree-to-license`.",
            ) from exc

        self._process = subprocess.Popen(
            ["arch", "-x86_64", str(helper_path), str(core_dir), rom_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        ready = self._readline()
        if ready != "READY":
            raise RosettaSnesError(
                f"Rosetta SNES helper failed to start: {ready or self._stderr_tail()}",
            )

    def close(self):
        if getattr(self, "_process", None) is None:
            return
        try:
            self._send_command("QUIT")
        except Exception:
            pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process = None

    def _stderr_tail(self):
        process = self._process
        if not process or process.stderr is None:
            return ""
        if process.poll() is None:
            return ""
        return process.stderr.read().decode("utf-8", errors="replace").strip()

    def _readline(self):
        if self._process.stdout is None:
            raise RosettaSnesError("Rosetta SNES helper stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            raise RosettaSnesError(
                "Rosetta SNES helper exited unexpectedly"
                + (f": {self._stderr_tail()}" if self._stderr_tail() else ""),
            )
        return line.decode("utf-8", errors="replace").rstrip("\n")

    def _readexactly(self, size):
        if self._process.stdout is None:
            raise RosettaSnesError("Rosetta SNES helper stdout is unavailable")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._process.stdout.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) != size:
            raise RosettaSnesError(
                "Rosetta SNES helper returned a short frame"
                + (f": {self._stderr_tail()}" if self._stderr_tail() else ""),
            )
        return bytes(chunks)

    def _send_command(self, command, payload=None):
        if self._process.stdin is None:
            raise RosettaSnesError("Rosetta SNES helper stdin is unavailable")
        self._process.stdin.write(command.encode("utf-8") + b"\n")
        if payload is not None:
            self._process.stdin.write(payload)
            self._process.stdin.write(b"\n")
        self._process.stdin.flush()
        response = self._readline()
        if response == "OK":
            return response
        if response.startswith("ERR "):
            raise RosettaSnesError(response[4:])
        raise RosettaSnesError(f"Unexpected helper response: {response}")

    def set_state(self, state):
        self._send_command(f"STATE {len(state)}", state)
        return True

    def set_button_mask(self, mask, player):
        flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
        value = 0
        for index, enabled in enumerate(flat):
            value |= (int(enabled) & 1) << index
        self._send_command(f"BUTTON {player} {value}")

    def step(self):
        self._send_command("STEP")

    def add_cheat(self, code):
        payload = code.encode("utf-8")
        self._send_command(f"ADD_CHEAT {len(payload)}", payload)

    def clear_cheats(self):
        self._send_command("CLEAR_CHEATS")

    def get_screen(self):
        if self._process.stdin is None:
            raise RosettaSnesError("Rosetta SNES helper stdin is unavailable")
        self._process.stdin.write(b"SCREEN\n")
        self._process.stdin.flush()
        header = self._readline()
        if header.startswith("ERR "):
            raise RosettaSnesError(header[4:])
        parts = header.split()
        if len(parts) != 3 or parts[0] != "SCREEN":
            raise RosettaSnesError(f"Unexpected helper response: {header}")
        width = int(parts[1])
        height = int(parts[2])
        frame = self._readexactly(width * height * 3)
        return np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3)).copy()

    def __del__(self):
        self.close()


class RosettaSnesEmulator:
    def __init__(self, rom_path):
        self.native_emulator = None
        self._helper = None
        self.native_emulator = NativeRetroEmulator(rom_path)
        try:
            self._helper = _RosettaSnesHelper(rom_path)
        except Exception:
            del self.native_emulator
            raise

    def step(self):
        self.native_emulator.step()
        self._helper.step()

    def set_button_mask(self, mask, player=0):
        self.native_emulator.set_button_mask(mask, player)
        self._helper.set_button_mask(mask, player)

    def get_state(self):
        return self.native_emulator.get_state()

    def set_state(self, state):
        native_ok = self.native_emulator.set_state(state)
        helper_ok = self._helper.set_state(state)
        return native_ok and helper_ok

    def get_screen(self):
        return self._helper.get_screen()

    def get_rotation(self):
        return self.native_emulator.get_rotation()

    def get_screen_rate(self):
        return self.native_emulator.get_screen_rate()

    def get_audio(self):
        return self.native_emulator.get_audio()

    def get_audio_rate(self):
        return self.native_emulator.get_audio_rate()

    def get_resolution(self):
        return self.native_emulator.get_resolution()

    def configure_data(self, data):
        self.native_emulator.configure_data(data)

    def add_cheat(self, code):
        self.native_emulator.add_cheat(code)
        self._helper.add_cheat(code)

    def clear_cheats(self):
        self.native_emulator.clear_cheats()
        self._helper.clear_cheats()

    def __getattr__(self, name):
        native = self.__dict__.get("native_emulator")
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)

    def __del__(self):
        helper = self.__dict__.get("_helper")
        if helper is not None:
            helper.close()

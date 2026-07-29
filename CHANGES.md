# Changelog

## Unreleased

* Nothing yet.

## 1.0.1.post37 - 2026-07-29

* default `RetroVecEnv(use_fire_reset=False)` to the neutral upstream
  `RetroEnv` reset behavior while retaining explicit Atari FIRE reset support

## 1.0.1.post36 - 2026-07-27

* add the immutable Turbo Vector API v1 declaration for capabilities, signals,
  action semantics, observation ownership, state catalogs, and per-lane RGB
  rendering
* make `render_lane()` return one raw lane, `get_images()` return every lane,
  and `render()` return lane zero

## 1.0.1.post35 - 2026-07-22

* let the interactive CLI launch any imported game ID with repeatable startup
  button presses, friendly power-on state selection, and Atari mode/difficulty
  configuration
* use the full controller action set in the interactive CLI so mapped console
  controls such as SELECT and RESET reach the emulator

## 1.0.1.post34 - 2026-07-20

* unify built-in, game-owned preset, and inline exact action tables under
  `use_restricted_actions`, with validated controller labels, semantic action
  meanings, deterministic hashes, and scalar/vector parity
* add reusable, per-lane live snapshot handles through
  `capture_snapshots(mask)` and mixed snapshot/catalog restoration through
  masked `reset()`, including exact cross-lane fan-out without advancing
  emulation
  * expose capability discovery and reject scripted scenarios or cores that
    cannot preserve exact state
* restore scalar `RetroEnv` to the upstream Stable Retro API and behavior
* make native `RetroVecEnv` permanently use disabled/manual autoreset
  * retain terminal observations until an explicit masked reset
  * reject stepping while any terminal lane remains pending reset
  * keep per-lane seeds and explicit `state_indices` selection from an immutable `state_catalog`
  * leave saved-state sampling policy and reset routing to the caller
* remove task events, `done_on`, same-step terminal payloads, and dynamic
  reset-policy mutation from the core `RetroVecEnv` API
* keep `RetroVecEnv` observation geometry stable when loading saved states that
  change the libretro core's active frame geometry
  * apply integration crops to each current frame instead of freezing cold-boot margins
  * reject native frame/observation shape mismatches before writing observation buffers

## 1.0.0

* add `EzPickle` support for `RetroEnv` to improve compatibility with multiprocessing/vectorized RL tooling
  * pickling recreates environments from constructor arguments
  * live emulator/viewer/movie runtime handles are not serialized

## 0.9.9

* fix human-mode rendering regression that could show a blank white window on some systems
* add four experimental RPG integrations
* clean up NHL94 environments
* documentation updates, including supported games table and Linux/macOS installation guides
* add paper/reference documentation

## 0.9.8

* fix packaging issue in `0.9.7` wheels where `data/stable` was empty

## 0.9.7

* add experimental Nintendo 64 core integration
* add experimental Nintendo DS core integration (MelonDS)
* add experimental Dreamcast core integration (Flycast)
* fix rendering/observations for vertical-screen games
* fix crash when integrating arcade games

### Breaking Changes (with backward compatibility)

**Package import name changed from `retro` to `stable_retro`**

- Users should now use `import stable_retro` instead of `import retro`
- The old `import retro` will continue to work with a deprecation warning for backward compatibility
- This change aligns the Python import name with the PyPI package name `stable-retro`
- All internal code has been updated to use `stable_retro`
- Documentation updated to reflect the new import name
- Backward compatibility will be maintained for multiple versions to allow gradual migration

## 0.9.6

* add FBNeo core support (arcade ROMs)
* fix retro/examples scripts for modern Gymnasium

## 0.9.5

* packaging/CI fixes for source distribution and build artifacts

## 0.9.4

* macOS build/CI support updates (including macos-14)
* build system dependency/version bumps for improved compatibility

## 0.9.3

* add Python 3.11 and 3.12 support (including manylinux wheels)
* fix build issues on Apple Silicon (M-series)

## 0.9.2

* build and publish manylinux wheels via cibuildwheel
* integration tool build fixes/workarounds

## 0.9.1

* add Apple Silicon (arm64) support
* add Windows build support for Python 3.10
* upgrade from Gym to Gymnasium
* add Sega 32X core support
* add Sega Saturn core support
* add Sega CD core support

## 0.9.0

* fix cores build on GCC 10
* add option to record interactive gameplay

## 0.8.0

* add python 3.8 support
* drop python 3.5 due to build issues on windows

## 0.7.1

* fix discrete and multi-discrete action space filtering
* fix random printfs when making environments
* data fixes for AeroStar-GameBoy, ChaseHQII-Genesis, Geimos-Nes, MagicalTaruruutoKun-Genesis,  KanshakudamaNageKantarouNoToukaidouGojuusanTsugi-Nes, TigerHeli-Nes (may change reward for these games, mostly these are bug fixes)
# python 3.5 compatibility fix (thanks @kieran-lemayai!)
* fix for new pyglet minor version that breaks backward compatibility (thanks @fsimond!)
* json parsing fix (thanks @eaplatanios!)
* minor memory leak fix (thanks @eaplatanios!)

## 0.7.0

* move some buggy games from the `stable` integrations folder to `experimental`
* minor bug fixes including fixes to a few game scenarios
* more docs
* add ability to use arbitrary additional integration directories
* integration UI searches for current Python's Gym Retro data directory
* import script can now accept files in addition to directories
* you can now use RAM observations by sending `obs_type=stable_retro.Observations.RAM` to `stable_retro.make`
* update Atari 2600 emulator

## 0.6.0

* add cores for GB/C, GBA, GG, NES, SMS, SNES, TurboGrafx
* add integration UI and searching
* add basic scenario access to Lua
* improve testing tooling
* multi-agent support
* cleaned up API:
  * everything involving data, e.g. game and state listing, file lookup and data path handling, has been moved into retro.data
  * importing retro.data.experiment or retro.data.contrib includes additional games and data that may not be as well-tested
  * retro.ACTIONS_* and retro.STATE_* have been replaced with retro.Actions.* and retro.State.* enums
  * retro.data.GameData objects no longer need an associated RetroEmulator object, though some functionality will not work
* add screen cropping
* added RetroEnv.get_action_meaning to describe the correlation between actions and buttons
* fixed d-pad action filtering so e.g. UP+DOWN+LEFT reduces to LEFT instead of NOOP
* add parallelism, lossless videos, info dict, disabling audio and numpy action dumping to playback_movies
* update LuaJIT to 2.1.0-beta3

## 0.5.6

* fix generating corrupted bk2s
* reliabilty fixes and minor enhancements to playback_movies

## 0.5.5

* allow Atari height to be different per game
* update pybind11 dependency
* add parallelism, lossless videos, info dict and numpy action dumping to playback_movies
* fix crashes with TensorFlow

## 0.5.4

* improved Windows support
* refreshed Python memory access
* build manylinux1-compatible wheels
* minor documentation fixes
* minor fixes to playback_movie script
* update Atari 2600 emulator

## 0.5.3

* fix Lua on Windows
* add Windows support to import.sega_classics
* only use system libzip if compatible

## 0.5.2

* initial public release

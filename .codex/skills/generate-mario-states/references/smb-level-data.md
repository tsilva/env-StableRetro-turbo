# Super Mario Bros NES Level Data

Use these facts when generating `SuperMarioBros-Nes-v0` states.

## Savestate Layout

In the local stable-retro-turbo NES savestate format, the first 2 KiB NES RAM block starts at byte offset `93` in the raw decompressed `.state` file.

State files are gzip-compressed native emulator savestates.

## RAM Addresses

The SMB disassembly labels used for state generation:

- `GameEngineSubroutine = $000e`
- `OperMode = $0770`
- `OperMode_Task = $0772`
- `BackgroundColorCtrl = $0744`
- `AreaType = $074e`
- `AreaPointer = $0750`
- `LevelNumber = $075c`
- `WorldNumber = $075f`
- `AreaNumber = $0760`
- `OffScr_LevelNumber = $0763`
- `OffScr_WorldNumber = $0766`
- `OffScr_AreaNumber = $0767`
- `SwimmingFlag = $0704`
- `AreaMusicQueue = $00fb`

To force a level load, patch the world, level, area number, area pointer, and offscreen mirrors; set `OperMode = 1`, `OperMode_Task = 0`, and `GameEngineSubroutine = 0`; clear stale entrance state (`EntrancePage`, `AltEntranceControl`, `HalfwayPage`, `OffScr_HalfwayPage`, `PlayerEntranceCtrl`); then let the emulator run until SMB's own `InitializeArea` and screen routines settle.

## Normal Level Area Map

Numbers are zero-based RAM values. `level` is `LevelNumber`; `area` is `AreaNumber`; `ptr` is `AreaPointer`.

| State | world | level | area | ptr |
| --- | ---: | ---: | ---: | ---: |
| Level1-1 | 0 | 0 | 0 | 0x25 |
| Level1-2 | 0 | 1 | 1 | 0x29 |
| Level1-3 | 0 | 2 | 3 | 0x26 |
| Level1-4 | 0 | 3 | 4 | 0x60 |
| Level2-1 | 1 | 0 | 0 | 0x28 |
| Level2-2 | 1 | 1 | 2 | 0x01 |
| Level2-3 | 1 | 2 | 3 | 0x27 |
| Level2-4 | 1 | 3 | 4 | 0x62 |
| Level3-1 | 2 | 0 | 0 | 0x24 |
| Level3-2 | 2 | 1 | 1 | 0x35 |
| Level3-3 | 2 | 2 | 2 | 0x20 |
| Level3-4 | 2 | 3 | 3 | 0x63 |
| Level4-1 | 3 | 0 | 0 | 0x22 |
| Level4-2 | 3 | 1 | 1 | 0x29 |
| Level4-3 | 3 | 2 | 3 | 0x2c |
| Level4-4 | 3 | 3 | 4 | 0x61 |
| Level5-1 | 4 | 0 | 0 | 0x2a |
| Level5-2 | 4 | 1 | 1 | 0x31 |
| Level5-3 | 4 | 2 | 2 | 0x26 |
| Level5-4 | 4 | 3 | 3 | 0x62 |
| Level6-1 | 5 | 0 | 0 | 0x2e |
| Level6-2 | 5 | 1 | 1 | 0x23 |
| Level6-3 | 5 | 2 | 2 | 0x2d |
| Level6-4 | 5 | 3 | 3 | 0x60 |
| Level7-1 | 6 | 0 | 0 | 0x33 |
| Level7-2 | 6 | 1 | 2 | 0x01 |
| Level7-3 | 6 | 2 | 3 | 0x27 |
| Level7-4 | 6 | 3 | 4 | 0x64 |
| Level8-1 | 7 | 0 | 0 | 0x30 |
| Level8-2 | 7 | 1 | 1 | 0x32 |
| Level8-3 | 7 | 2 | 2 | 0x21 |
| Level8-4 | 7 | 3 | 3 | 0x65 |

The area pointer's high bits encode area type:

- `0`: water
- `1`: ground/platform
- `2`: underground
- `3`: castle

## Level 1-2 Special Case

`Level1-2` starts on the aboveground pipe-entry strip (`AreaNumber = 1`, `AreaPointer = 0x29`) shown at the top-left of the NESMaps full map. SMB then automatically transitions into the cave (`AreaNumber = 2`, `AreaPointer = 0xc0`).

For stable-retro training states, use the settled cave start rather than the coin bonus room or the transient pipe-entry strip:

- Patch initial values from the table: `world=0`, `level=1`, `area=1`, `ptr=0x29`.
- Clear the stale entrance variables listed above.
- Use `Level1-1.state` as the base state and run about `540` frames.
- Validate final RAM as `WorldNumber=0`, `LevelNumber=1`, `AreaNumber=2`, `AreaPointer=0xc0`, `OperMode=1`, `OperMode_Task>=3`.
- Compare the screenshot to the `y=240` cave row of `SuperMarioBrosMap1-2.png`.

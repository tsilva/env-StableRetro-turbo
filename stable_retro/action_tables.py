"""Resolve built-in and integration-owned Stable Retro action contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from stable_retro.enums import Actions


RESERVED_ACTION_SET_NAMES = frozenset(
    {"all", "filtered", "discrete", "multi_discrete"}
)

PlayerAction: TypeAlias = Sequence[str]
JointAction: TypeAlias = Sequence[PlayerAction]
ActionTable: TypeAlias = Sequence[PlayerAction | JointAction]


@dataclass(frozen=True)
class ActionSpec:
    """Normalized action-space request used by scalar and vector environments."""

    mode: str
    builtin: Actions | None
    preset: str | None
    table: tuple[Any, ...] | None
    meanings: tuple[str, ...] | None
    masks: tuple[tuple[int, ...], ...] | None
    table_hash: str | None


def _metadata(game: str, inttype: Any) -> Mapping[str, Any]:
    import stable_retro.data as retro_data

    path = retro_data.get_file_path(game, "metadata.json", inttype)
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load action metadata for {game!r}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"metadata for {game!r} must be a JSON object")
    return value


def load_action_sets(game: str, inttype: Any) -> dict[str, tuple[str, Any]]:
    """Load a game's case-insensitive named action tables from metadata.json."""
    metadata = _metadata(game, inttype)
    raw = metadata.get("action_sets")
    if raw is None:
        # RETRO_DATA_PATH commonly points at a user-owned ROM integration copied
        # by an older release. Keep executable ROM/state files there while using
        # the package's current Stable integration metadata for new declarations.
        packaged = Path(__file__).parent / "data" / "stable" / game / "metadata.json"
        if packaged.is_file():
            try:
                packaged_metadata = json.loads(packaged.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"could not load packaged action metadata for {game!r}: {exc}"
                ) from exc
            raw = packaged_metadata.get("action_sets")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"metadata action_sets for {game!r} must be an object")
    result: dict[str, tuple[str, Any]] = {}
    for name, table in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"metadata action_sets for {game!r} has an invalid name")
        folded = name.casefold()
        if folded in RESERVED_ACTION_SET_NAMES:
            raise ValueError(
                f"metadata action set {name!r} for {game!r} collides with a built-in mode"
            )
        if folded in result:
            raise ValueError(
                f"metadata action_sets for {game!r} contains case-insensitive duplicate {name!r}"
            )
        result[folded] = (name, table)
    return result


def _builtin(value: Any) -> Actions | None:
    if isinstance(value, Actions):
        return value
    if isinstance(value, str):
        key = value.strip().casefold()
        if key in RESERVED_ACTION_SET_NAMES:
            return Actions[key.upper()]
    return None


def _sequence(value: Any, message: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(message)
    return value


def _player_buttons(
    raw: Any,
    *,
    button_to_index: Mapping[str, int],
    context: str,
) -> tuple[tuple[str, ...], int]:
    values = _sequence(raw, f"{context} must be a list of button labels")
    labels: list[str] = []
    seen: set[str] = set()
    mask = 0
    for label in values:
        if not isinstance(label, str):
            raise ValueError(f"{context} button labels must be strings")
        if label in seen:
            raise ValueError(f"{context} contains duplicate button label {label!r}")
        try:
            index = button_to_index[label]
        except KeyError as exc:
            valid = ", ".join(repr(name) for name in button_to_index)
            raise ValueError(
                f"{context} contains unknown button {label!r}; valid labels: {valid}"
            ) from exc
        labels.append(label)
        seen.add(label)
        mask |= 1 << index
    return tuple(labels), mask


def _player_meaning(labels: tuple[str, ...]) -> str:
    return "noop" if not labels else "_".join(label.lower() for label in labels)


def normalize_action_table(
    table: Any,
    *,
    buttons: Sequence[str | None],
    players: int,
    context: str = "action table",
) -> tuple[
    tuple[Any, ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    str,
]:
    """Validate a table and return its public form, meanings, masks, and hash."""
    if players <= 0:
        raise ValueError("players must be positive")
    actions = _sequence(table, f"{context} must be a non-empty list of actions")
    if not actions:
        raise ValueError(f"{context} must contain at least one action")
    button_to_index = {
        label: index for index, label in enumerate(buttons) if label is not None
    }
    normalized: list[Any] = []
    meanings: list[str] = []
    masks: list[tuple[int, ...]] = []
    seen_masks: set[tuple[int, ...]] = set()
    for action_index, raw_action in enumerate(actions):
        if players == 1:
            labels, mask = _player_buttons(
                raw_action,
                button_to_index=button_to_index,
                context=f"{context} action {action_index}",
            )
            public_action: Any = labels
            action_masks = (mask,)
            meaning = _player_meaning(labels)
        else:
            player_actions = _sequence(
                raw_action,
                f"{context} action {action_index} must contain one action per player",
            )
            if len(player_actions) != players:
                raise ValueError(
                    f"{context} action {action_index} must contain exactly {players} player actions"
                )
            public_players: list[tuple[str, ...]] = []
            player_masks: list[int] = []
            player_meanings: list[str] = []
            for player_index, raw_player_action in enumerate(player_actions):
                labels, mask = _player_buttons(
                    raw_player_action,
                    button_to_index=button_to_index,
                    context=(
                        f"{context} action {action_index} player {player_index + 1}"
                    ),
                )
                public_players.append(labels)
                player_masks.append(mask)
                player_meanings.append(
                    f"p{player_index + 1}_{_player_meaning(labels)}"
                )
            public_action = tuple(public_players)
            action_masks = tuple(player_masks)
            meaning = "__".join(player_meanings)
        if action_masks in seen_masks:
            raise ValueError(
                f"{context} action {action_index} duplicates an earlier controller action"
            )
        normalized.append(public_action)
        meanings.append(meaning)
        masks.append(action_masks)
        seen_masks.add(action_masks)
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    table_hash = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return tuple(normalized), tuple(meanings), tuple(masks), table_hash


def resolve_action_spec(
    value: Any,
    *,
    game: str,
    inttype: Any,
    buttons: Sequence[str | None],
    players: int,
) -> ActionSpec:
    """Resolve an Actions mode, metadata preset name, or inline exact table."""
    builtin = _builtin(value)
    if builtin is not None:
        return ActionSpec(
            mode=builtin.name.lower(),
            builtin=builtin,
            preset=None,
            table=None,
            meanings=None,
            masks=None,
            table_hash=None,
        )
    preset = None
    table = value
    if isinstance(value, str):
        action_sets = load_action_sets(game, inttype)
        try:
            preset, table = action_sets[value.strip().casefold()]
        except KeyError as exc:
            valid = sorted(RESERVED_ACTION_SET_NAMES | set(action_sets))
            raise ValueError(
                f"unknown use_restricted_actions value {value!r} for {game!r}; "
                f"valid values: {', '.join(valid)}"
            ) from exc
    normalized, meanings, masks, table_hash = normalize_action_table(
        table,
        buttons=buttons,
        players=players,
        context=(f"action set {preset!r}" if preset is not None else "action table"),
    )
    return ActionSpec(
        mode="custom_discrete",
        builtin=None,
        preset=preset,
        table=normalized,
        meanings=meanings,
        masks=masks,
        table_hash=table_hash,
    )


__all__ = [
    "ActionTable",
    "ActionSpec",
    "RESERVED_ACTION_SET_NAMES",
    "load_action_sets",
    "normalize_action_table",
    "resolve_action_spec",
]

import json

import pytest

from stable_retro.action_tables import normalize_action_table, resolve_action_spec
from stable_retro.enums import Actions


NES_BUTTONS = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")


def test_builtin_string_modes_normalize_to_actions():
    spec = resolve_action_spec(
        "multi_discrete",
        game="unused",
        inttype=None,
        buttons=NES_BUTTONS,
        players=1,
    )

    assert spec.mode == "multi_discrete"
    assert spec.builtin is Actions.MULTI_DISCRETE
    assert spec.table is None


def test_inline_table_preserves_order_and_hashes_semantic_masks():
    first = normalize_action_table(
        [[], ["RIGHT", "A"]],
        buttons=NES_BUTTONS,
        players=1,
    )
    reordered_buttons = normalize_action_table(
        [[], ["A", "RIGHT"]],
        buttons=NES_BUTTONS,
        players=1,
    )

    assert first[0] == ((), ("RIGHT", "A"))
    assert first[1] == ("noop", "right_a")
    assert reordered_buttons[1] == ("noop", "a_right")
    assert first[3] == reordered_buttons[3]


def test_multiplayer_table_is_joint_and_not_a_cartesian_product():
    table, meanings, masks, _ = normalize_action_table(
        [[["LEFT"], ["A"]], [[], []]],
        buttons=NES_BUTTONS,
        players=2,
    )

    assert len(table) == 2
    assert meanings == ("p1_left__p2_a", "p1_noop__p2_noop")
    assert masks == ((1 << 6, 1 << 8), (0, 0))


@pytest.mark.parametrize(
    "table, message",
    [
        ([], "at least one action"),
        ([["RIGHT"], ["RIGHT"]], "duplicates"),
        ([["RIGHT", "RIGHT"]], "duplicate button"),
        ([["MISSING"]], "unknown button"),
    ],
)
def test_invalid_inline_tables_are_rejected(table, message):
    with pytest.raises(ValueError, match=message):
        normalize_action_table(table, buttons=NES_BUTTONS, players=1)


def test_metadata_preset_resolution(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"action_sets": {"Simple": [[], ["RIGHT"], ["A"]]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stable_retro.data.get_file_path",
        lambda *_args, **_kwargs: str(metadata),
    )

    spec = resolve_action_spec(
        "simple",
        game="Example-Nes-v0",
        inttype=None,
        buttons=NES_BUTTONS,
        players=1,
    )

    assert spec.mode == "custom_discrete"
    assert spec.preset == "Simple"
    assert spec.meanings == ("noop", "right", "a")


def test_unknown_preset_lists_builtins_and_game_presets(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"action_sets": {"simple": [[]]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stable_retro.data.get_file_path",
        lambda *_args, **_kwargs: str(metadata),
    )

    with pytest.raises(ValueError, match="all.*simple"):
        resolve_action_spec(
            "missing",
            game="Example-Nes-v0",
            inttype=None,
            buttons=NES_BUTTONS,
            players=1,
        )


def test_metadata_cannot_shadow_builtin_mode(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"action_sets": {"ALL": [[]]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stable_retro.data.get_file_path",
        lambda *_args, **_kwargs: str(metadata),
    )

    with pytest.raises(ValueError, match="collides with a built-in"):
        resolve_action_spec(
            "missing",
            game="Example-Nes-v0",
            inttype=None,
            buttons=NES_BUTTONS,
            players=1,
        )

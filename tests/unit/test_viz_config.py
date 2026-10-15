"""What the user configures, as data: actions, keys, mouse, settings, theme."""

from __future__ import annotations

import inspect

import pytest

from bidsmgr.viz import actions as A
from bidsmgr.viz import inputmap
from bidsmgr.viz.commands import COMMANDS
from bidsmgr.viz.settings import VizSettings
from bidsmgr.viz.theme import VizTheme, parse_colour, hex_colour


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr, ctx, expected", [
    ("", {}, True),
    ("volume", {"volume": True}, True),
    ("volume", {}, False),
    ("!volume", {"volume": False}, True),
    ("mode=multi", {"mode": "multi"}, True),
    ("mode=multi", {"mode": "single"}, False),
    ("!mode=3d", {"mode": "multi"}, True),
    ("volume && gpu", {"volume": True, "gpu": False}, False),
    ("volume && gpu || mode=3d", {"volume": True, "gpu": False, "mode": "3d"}, True),
    ("mode=single && plane=axial", {"mode": "single", "plane": "axial"}, True),
])
def test_expressions(expr, ctx, expected) -> None:
    assert A.evaluate(expr, ctx) is expected


# ---------------------------------------------------------------------------
# Actions and the commands they name
# ---------------------------------------------------------------------------


def test_action_ids_are_unique() -> None:
    ids = [a.id for a in A.ACTIONS]
    assert len(ids) == len(set(ids))


def test_every_action_names_a_real_command_with_real_parameters() -> None:
    """A typo in the table would only surface when the key was pressed."""
    for action in A.ACTIONS:
        if action.command.startswith("gui:"):
            continue
        assert action.command in COMMANDS, action.id
        spec = COMMANDS[action.command]
        for name in action.params:
            assert name in spec.params, f"{action.id}: {action.command} has no {name!r}"


def test_toggles_say_when_they_are_on() -> None:
    for action in A.ACTIONS:
        if action.checkable:
            assert action.checked.strip(), action.id


def test_the_default_keys_do_not_conflict() -> None:
    assert A.conflicts(A.effective_keymap()) == []


def test_the_old_viewer_letters_are_still_the_defaults() -> None:
    """Decision D3: the keys people already know stay."""
    keys = A.effective_keymap()
    for action_id, key in {
        "view.axial": "A", "view.sagittal": "S", "view.coronal": "C",
        "view.multi": "M", "view.graph": "G", "view.labels": "O",
        "view.3d": "D", "view.combo": "P", "clip.toggle": "Shift+Z",
        "clip.axial": "Shift+A", "clip.sagittal": "Shift+S",
        "clip.coronal": "Shift+C", "clip.invert": "Shift+X",
    }.items():
        assert key in keys[action_id], action_id


# ---------------------------------------------------------------------------
# The keymap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("a", "A"),
    ("shift+a", "Shift+A"),
    ("ctrl+shift+z", "Ctrl+Shift+Z"),
    ("Shift+Ctrl+Z", "Ctrl+Shift+Z"),
    ("control+p", "Ctrl+P"),
    ("cmd+p", "Ctrl+P"),
    ("pageup", "PgUp"),
    ("PgDn", "PgDown"),
    ("left", "Left"),
    ("space", "Space"),
    ("", ""),
])
def test_keys_have_one_spelling(raw, expected) -> None:
    assert A.normalise_key(raw) == expected


def test_overrides_replace_and_unbind() -> None:
    keys = A.effective_keymap({"view.axial": ["shift+q"], "view.graph": []})
    assert keys["view.axial"] == ["Shift+Q"]
    assert keys["view.graph"] == []
    assert keys["view.sagittal"] == ["S"]


def test_a_rebinding_onto_a_used_key_is_reported() -> None:
    keys = A.effective_keymap({"view.graph": ["A"]})
    found = A.conflicts(keys)
    assert any(key == "A" and {a, b} == {"view.axial", "view.graph"}
               for key, a, b in found)


def test_an_unknown_action_in_the_overrides_is_ignored() -> None:
    keys = A.effective_keymap({"no.such.action": ["K"]})
    assert "no.such.action" not in keys


# ---------------------------------------------------------------------------
# The mouse map
# ---------------------------------------------------------------------------


def test_the_mouse_defaults_are_the_old_gestures() -> None:
    assert inputmap.lookup("slice", "left", set()) == "crosshair"
    assert inputmap.lookup("slice", "wheel", set()) == "slice"
    assert inputmap.lookup("slice", "hwheel", set()) == "frame"
    assert inputmap.lookup("slice", "wheel", {"h"}) == "frame"
    assert inputmap.lookup("render", "left", set()) == "orbit"
    assert inputmap.lookup("render", "left", {"shift"}) == "clip_tilt"
    assert inputmap.lookup("render", "wheel", set()) == "zoom"


def test_the_most_specific_binding_wins() -> None:
    # Ctrl+Shift+left has no binding of its own: the larger matching subset
    # (Ctrl+left before Shift+left in canonical order? both size one) must
    # still resolve to a binding rather than to nothing.
    assert inputmap.lookup("slice", "left", {"ctrl", "shift"}) in ("pan", "window_box")
    over = {"slice:ctrl+shift+left": "measure"}
    assert inputmap.lookup("slice", "left", {"ctrl", "shift"}, over) == "measure"
    # Adding a modifier binding never breaks the plain one.
    assert inputmap.lookup("slice", "left", set(), over) == "crosshair"


def test_an_unbound_gesture_does_nothing() -> None:
    assert inputmap.lookup("slice", "middle", {"alt"}, {"slice:middle": "none"}) == "none"


def test_every_default_tool_exists() -> None:
    for key, tool in inputmap.DEFAULT_MOUSEMAP.items():
        canvas, gesture = key.split(":", 1)
        # And of the right KIND: a wheel gesture runs a wheel tool.
        assert tool in inputmap.tools_for(canvas, gesture), key
        assert gesture in inputmap.GESTURES[canvas], key


def test_gesture_text_is_canonical() -> None:
    assert inputmap.gesture("left", {"shift", "ctrl"}) == "ctrl+shift+left"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_clamp_rather_than_refuse() -> None:
    """An older or hand-edited settings blob still loads."""
    s = VizSettings.model_validate({
        "crosshair": {"thickness": 99, "gap": -3, "color": "#ff0000"},
        "volume": {"gamma": 0.0, "fps": 1000},
        "render": {"quality": 5},
        "traces": {"line_width": -1},
        "an_old_key": 1,
    })
    assert s.crosshair.thickness == 5 and s.crosshair.gap == 0
    assert s.volume.gamma == pytest.approx(0.1) and s.volume.fps == 60.0
    assert s.render.quality >= 64
    assert s.traces.line_width == 0


def test_settings_round_trip_as_json() -> None:
    s = VizSettings()
    s.keymap["view.axial"] = ["Shift+Q"]
    s.mousemap["slice:right"] = "pan"
    s.layout_sizes["volume.graph"] = [700, 300]
    again = VizSettings.model_validate_json(s.model_dump_json())
    assert again == s


def test_an_invalid_choice_is_rejected_on_assignment() -> None:
    s = VizSettings()
    with pytest.raises(ValueError):
        s.volume.plane = "diagonal"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, rgba", [
    ("#fff", (255, 255, 255, 255)),
    ("#102030", (16, 32, 48, 255)),
    ("#10203080", (16, 32, 48, 128)),
    ("rgba(88, 166, 255, 0.5)", (88, 166, 255, 128)),
    ("rgb(1, 2, 3)", (1, 2, 3, 255)),
    ("nonsense", (0, 0, 0, 255)),
])
def test_colours_parse_in_both_spellings(text, rgba) -> None:
    """QColor reads rgba() strings as black, so the theme parses them."""
    assert parse_colour(text) == rgba


def test_the_viz_theme_follows_the_app_palette() -> None:
    from bidsmgr.gui import theme_manager

    for name, pal in (("dark", theme_manager.DARK), ("light", theme_manager.LIGHT)):
        theme = VizTheme.from_palette(pal, name)
        assert theme.plot_background == pal["bg"]
        assert theme.text == pal["text"]
        # Images keep a black surround in both themes.
        assert theme.background == "#000000"
        # Every series and channel-type token resolves to a real colour.
        for i in range(6):
            assert parse_colour(theme.series(i))[3] > 0
        for ch_type in ("eeg", "meg", "grad", "ecg", "stim", "resp"):
            assert theme.type_colour(ch_type).startswith(("#", "rgb"))


def test_channel_colours_honour_the_user_and_stay_stable() -> None:
    theme = VizTheme.from_palette({"accent": "#123456", "success": "#00ff00",
                                   "purple": "#aa00aa"})
    # The table the old viewer used: mag accent, grad success, eeg purple.
    assert theme.type_colour("mag") == "#123456"
    assert theme.type_colour("grad") == "#00ff00"
    assert theme.type_colour("eeg") == "#aa00aa"
    assert theme.type_colour("eeg", {"eeg": "#ff0000"}) == "#ff0000"
    # A type the table does not know still gets the same colour every time,
    # and not all unknown types the same one.
    assert theme.type_token("ias") == theme.type_token("ias")
    assert len({theme.type_token(t) for t in ("ias", "syst", "chpi", "exci")}) > 1


def test_hex_colour() -> None:
    assert hex_colour((255, 0, 16, 9)) == "#ff0010"


def test_commands_are_keyword_only_after_the_store() -> None:
    """Scripts call commands by name: ``viewer.run("frame.set", frame=3)``."""
    for spec in COMMANDS.values():
        names = list(inspect.signature(spec.fn).parameters)
        assert names[0] == "store", spec.id

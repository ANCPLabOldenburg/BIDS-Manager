"""Keys and gestures are written the way the running OS names them.

The keymap stores Qt's portable spelling everywhere (``Ctrl+Shift+Z``); Qt
binds ``Ctrl`` to Command on macOS, so a label saying "Ctrl" sent a Mac user
to the wrong key (user report, 2026-10-07)."""

from __future__ import annotations

import pytest

from bidsmgr.viz import actions as A
from bidsmgr.viz import inputmap, keynames


@pytest.mark.parametrize("stored", sorted({A.normalise_key(k) for d in A.ACTIONS for k in d.keys}))
def test_windows_and_linux_keep_the_stored_spelling(stored) -> None:
    assert keynames.key(stored, mac=False) == stored


@pytest.mark.parametrize("stored, mac", [
    ("Ctrl+G", "⌘G"),
    ("Ctrl+Shift+Z", "⇧⌘Z"),
    ("Shift+A", "⇧A"),
    ("Alt+Left", "⌥←"),
    ("Alt+PgUp", "⌥ Page Up"),
    ("PgDown", "Page Down"),
    ("Backspace", "⌫"),
    ("Space", "Space"),
    ("Meta+Alt+Shift+Ctrl+K", "⌃⌥⇧⌘K"),     # Apple's order, whatever the stored one
    ("Ctrl++", "⌘+"),
    ("G", "G"),
    ("[", "["),
])
def test_a_mac_writes_apples_symbols(stored, mac) -> None:
    assert keynames.key(stored, mac=True) == mac


def test_no_default_key_says_ctrl_or_alt_on_a_mac() -> None:
    for d in A.ACTIONS:
        for k in d.keys:
            label = keynames.key(A.normalise_key(k), mac=True)
            assert "Ctrl" not in label and "Alt" not in label, (d.id, label)


@pytest.mark.parametrize("gesture, pc, mac", [
    ("left", "Click / drag", "Click / drag"),
    ("ctrl+left", "Ctrl + Click / drag", "⌘ + Click / drag"),
    ("alt+left", "Alt + Click / drag", "⌥ + Click / drag"),
    ("shift+wheel", "Shift + Scroll", "⇧ + Scroll"),
    ("ctrl+shift+wheel", "Ctrl + Shift + Scroll", "⇧⌘ + Scroll"),
    ("h+wheel", "Hold H + Scroll", "Hold H + Scroll"),
    ("hwheel", "Horizontal scroll", "Horizontal scroll"),
])
def test_gestures(gesture, pc, mac) -> None:
    assert keynames.gesture(gesture, mac=False) == pc
    assert keynames.gesture(gesture, mac=True) == mac


def test_every_gesture_of_the_mouse_map_has_a_mac_label_without_ctrl() -> None:
    for canvas, gestures in inputmap.GESTURES.items():
        for g in gestures:
            label = keynames.gesture(g, mac=True)
            assert "Ctrl" not in label and "Alt" not in label, (canvas, g, label)


def test_prose() -> None:
    assert keynames.mouse(["ctrl"], "drag", mac=False) == "Ctrl+drag"
    assert keynames.mouse(["ctrl", "shift"], "drag", mac=False) == "Ctrl+Shift+drag"
    assert keynames.mouse(["ctrl"], "drag", mac=True) == "⌘+drag"
    assert keynames.mouse(["ctrl", "shift"], "drag", mac=True) == "⇧⌘+drag"
    assert keynames.mouse([], "drag", mac=True) == "drag"


@pytest.mark.parametrize("needle", ["cmd", "command", "shift", "z"])
def test_a_search_finds_a_symbol_by_its_name(needle) -> None:
    assert needle in keynames.search_words(keynames.key("Ctrl+Shift+Z", mac=True))


def test_the_platform_default_is_the_running_one(monkeypatch) -> None:
    monkeypatch.setattr(keynames.sys, "platform", "darwin")
    assert keynames.key("Ctrl+I") == "⌘I"
    monkeypatch.setattr(keynames.sys, "platform", "win32")
    assert keynames.key("Ctrl+I") == "Ctrl+I"
    monkeypatch.setattr(keynames.sys, "platform", "linux")
    assert keynames.key("Ctrl+I") == "Ctrl+I"

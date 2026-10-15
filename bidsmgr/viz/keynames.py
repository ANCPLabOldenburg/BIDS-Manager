"""How a key or a mouse gesture is WRITTEN for the user, on this OS.

The keymap stores one portable spelling on every platform (``Ctrl+Shift+Z``,
``Alt+Left``), Qt's ``PortableText``, so a keymap exported on a Mac works on
Windows. Qt binds ``Ctrl`` to the Command key on macOS (and the Control key
to ``Meta``), so a Mac user pressing Command+Z IS pressing ``Ctrl+Z``, and
the canvases take a Command-drag as their ``ctrl+`` gesture. The viewers'
menus already say so (Qt writes ``⌘Z`` there). Everything else that named a
key (the shortcut window, the settings table, tooltips, help text) printed
the stored spelling, "Ctrl", which is the wrong key on a Mac.

This module is the one place that turns the stored spelling into a label:

* Windows and Linux: the stored spelling, unchanged (``Ctrl+Shift+Z``,
  ``Ctrl + Click / drag``).
* macOS: Apple's modifier symbols in Apple's order (⌃ ⌥ ⇧ ⌘) followed by the
  key, as macOS menus write them (``⇧⌘Z``, ``⌥←``); arrows as arrows, a
  named key in words (``⌥ Page Up``). A gesture keeps its words and swaps
  the modifiers (``⌘ + Click / drag``).

Never write a key name into a label by hand: call ``key``, ``mouse`` or
``gesture``. Pure functions; ``mac`` defaults to the running platform and is
a parameter so every platform's text can be tested on any one.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional

#: modifier -> (Windows and Linux, macOS symbol, macOS word)
_MODS = {
    "ctrl": ("Ctrl", "⌘", "Cmd"),
    "meta": ("Meta", "⌃", "Control"),
    "alt": ("Alt", "⌥", "Option"),
    "shift": ("Shift", "⇧", "Shift"),
}
#: The order each platform writes them in (``actions.normalise_key``'s, and
#: Apple's).
_ORDER = ("ctrl", "meta", "alt", "shift")
_ORDER_MAC = ("meta", "alt", "shift", "ctrl")
#: Other spellings a stored or typed key may use.
_ALIASES = {"control": "ctrl", "cmd": "ctrl", "command": "ctrl", "option": "alt",
            "opt": "alt"}

#: Keys a Mac writes differently: arrows and the two deletes as the symbols on
#: the keys, the rest in words (a Mac laptop has no Page Up key, and few
#: people read the menu symbol for it).
_MAC_KEYS = {
    "Left": "←", "Right": "→", "Up": "↑", "Down": "↓",
    "Backspace": "⌫", "Delete": "⌦", "Del": "⌦",
    "Return": "↩", "Enter": "⌤", "Tab": "⇥",
    "PgUp": "Page Up", "PgDown": "Page Down", "Esc": "Esc", "Escape": "Esc",
}

#: What a gesture's button or wheel is called in the help and the settings.
_GESTURES = {
    "left": "Click / drag", "right": "Right-drag", "middle": "Middle-drag",
    "wheel": "Scroll", "hwheel": "Horizontal scroll",
}

#: Words a search should find a symbol by (``cmd`` finds ``⇧⌘Z``).
_WORDS = {
    "⌘": "cmd command ctrl", "⌃": "control ctrl", "⌥": "option opt alt",
    "⇧": "shift", "←": "left", "→": "right", "↑": "up",
    "↓": "down", "⌫": "backspace delete", "⌦": "delete del",
    "↩": "return enter", "⌤": "enter", "⇥": "tab",
}


def is_mac() -> bool:
    return sys.platform == "darwin"


def _on_mac(mac: Optional[bool]) -> bool:
    return is_mac() if mac is None else bool(mac)


def _canonical(mods: Iterable[str]) -> set[str]:
    out = set()
    for m in mods:
        name = m.strip().lower()
        name = _ALIASES.get(name, name)
        if name in _MODS:
            out.add(name)
    return out


def modifiers(mods: Iterable[str], *, mac: Optional[bool] = None) -> str:
    """``["ctrl", "shift"]`` as this OS writes it: ``Ctrl+Shift``, or
    ``⇧⌘`` on a Mac."""
    names = _canonical(mods)
    if _on_mac(mac):
        return "".join(_MODS[m][1] for m in _ORDER_MAC if m in names)
    return "+".join(_MODS[m][0] for m in _ORDER if m in names)


def _split(seq: str) -> tuple[list[str], str]:
    """``Ctrl+Shift+Z`` -> (["Ctrl", "Shift"], "Z"); a ``+`` key survives."""
    if seq.endswith("++"):
        return [p for p in seq[:-2].split("+") if p], "+"
    if seq == "+":
        return [], "+"
    parts = seq.split("+")
    return parts[:-1], parts[-1]


def key(seq: str, *, mac: Optional[bool] = None) -> str:
    """A stored key sequence (``Ctrl+Shift+Z``) as this OS writes it:
    unchanged on Windows and Linux, ``⇧⌘Z`` on a Mac."""
    seq = (seq or "").strip()
    if not seq or not _on_mac(mac):
        return seq
    mods, base = _split(seq)
    shown = _MAC_KEYS.get(base, base)
    prefix = modifiers(mods, mac=True)
    if prefix and len(shown) > 1:
        return f"{prefix} {shown}"      # ⌥ Page Up, ⇧ Space
    return prefix + shown


def keys(seqs: Iterable[str], *, mac: Optional[bool] = None) -> list[str]:
    return [key(s, mac=mac) for s in seqs]


def mouse(mods: Iterable[str], action: str, *, spaced: bool = False,
          mac: Optional[bool] = None) -> str:
    """A modifier held with a mouse action: ``Ctrl+drag`` (``Ctrl + Click /
    drag`` when ``spaced``), ``⌘+drag`` on a Mac."""
    held = modifiers(mods, mac=mac)
    if not held:
        return action
    sep = " + " if spaced else "+"
    if _on_mac(mac):
        return f"{held}{sep}{action}"
    return sep.join(held.split("+") + [action])


def gesture(gesture_id: str, *, mac: Optional[bool] = None) -> str:
    """A mouse-map gesture (``ctrl+left``, ``h+wheel``) in words:
    ``Ctrl + Click / drag``, ``Hold H + Scroll``; ``⌘ + Click / drag`` on a
    Mac."""
    parts = gesture_id.split("+")
    base = _GESTURES.get(parts[-1], parts[-1])
    if "h" in parts[:-1]:
        base = f"Hold H + {base}"
    return mouse([p for p in parts[:-1] if p != "h"], base, spaced=True, mac=mac)


def search_words(label: str) -> str:
    """``label`` plus the words for its symbols, lower case, so a search for
    ``cmd`` or ``shift`` finds ``⇧⌘Z``."""
    extra = [w for ch, w in _WORDS.items() if ch in label]
    return " ".join([label.lower()] + extra)


__all__ = ["gesture", "is_mac", "key", "keys", "modifiers", "mouse", "search_words"]

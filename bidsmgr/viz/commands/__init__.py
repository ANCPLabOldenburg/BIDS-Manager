"""Every capability of the library, as a named command.

A command is a function ``fn(store, **params) -> set[str]`` registered under
an id such as ``"cursor.step"``. It changes the scene in place and returns
the paths it touched (``{"cursor"}``, ``{"layer:abc.display"}``), which is
how canvases know whether they have to redraw.

One path for every change is the point of the library. Buttons, menus,
keyboard shortcuts, the mouse, scripts and a linked second viewer all run the
same commands, so binding a key to anything is a line of keymap data, undo
works on the scene, and nothing can change the view behind the scene's back.

Parameters are validated by pydantic from the function's signature, so a
script that passes ``frame="7"`` gets a 7 and one that passes ``frame="x"``
gets an error naming the command.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ConfigDict, validate_call


@dataclass(frozen=True)
class CommandSpec:
    """What a command is, for menus, the palette, the keymap and scripts."""

    id: str
    title: str
    category: str
    fn: Callable[..., Any]
    undoable: bool = False
    help: str = ""
    params: tuple[str, ...] = field(default_factory=tuple)


COMMANDS: dict[str, CommandSpec] = {}


def command(
    command_id: str,
    title: str,
    *,
    category: str = "General",
    undoable: bool = False,
    help: str = "",
):
    """Register ``fn`` as command ``command_id``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if command_id in COMMANDS:
            raise ValueError(f"command {command_id!r} registered twice")
        params = list(inspect.signature(fn).parameters)
        # The first argument is the store, which is not user input and is
        # only named for type checkers; validate everything after it.
        original = dict(fn.__annotations__)
        fn.__annotations__ = {**original, params[0]: Any}
        try:
            validated = validate_call(
                fn, config=ConfigDict(arbitrary_types_allowed=True),
            )
        finally:
            fn.__annotations__ = original
        names = tuple(params[1:])
        COMMANDS[command_id] = CommandSpec(
            id=command_id, title=title, category=category, fn=validated,
            undoable=undoable, help=help or (fn.__doc__ or "").strip().split("\n")[0],
            params=names,
        )
        return fn

    return decorate


def spec(command_id: str) -> CommandSpec:
    try:
        return COMMANDS[command_id]
    except KeyError:
        raise KeyError(f"unknown command {command_id!r}") from None


def _load_all() -> None:
    # Importing registers. Kept here so `import bidsmgr.viz.commands` is
    # enough to see every command.
    from . import volume  # noqa: F401
    from . import render  # noqa: F401
    from . import signal  # noqa: F401
    from . import spectrum  # noqa: F401


_load_all()

__all__ = ["COMMANDS", "CommandSpec", "command", "spec"]

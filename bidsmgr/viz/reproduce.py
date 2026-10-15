"""The command line of a view: what reproduces it from a shell or a script.

FSLeyes calls it "show command line"; NiiVue's equivalent is a document of
its options. Here the view is a scene, and every change to a scene is a
command, so a view is reproduced by opening the same files and running the
commands that set what differs from how they open:

* :func:`shell` writes a ``bidsmgr-view`` call, to open the view in a
  window or (``render=...``) to write it as a PNG with no window at all;
* :func:`python` writes the same as a script over
  :mod:`bidsmgr.viz.render2d`, for a tool that embeds the library.

Qt-free. A layer computed in the viewer (a quality map) has no file to
open, and is named in a comment rather than silently dropped.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Literal, Optional

from .scene import VolumeDisplay

#: The look of a layer a command can set (``layer.set``), in its order.
DISPLAY_FIELDS = ("colormap", "colormap_negative", "window", "window_negative", "gamma",
                  "invert", "opacity", "threshold_mode", "outline_px", "interpolation")
#: The 2-D planes each layout shows, for a figure rendered without a window.
LAYOUT_PLANES = {"multi": ("sagittal", "coronal", "axial"), "combo": ("sagittal", "coronal", "axial"),
                 "hero": ("sagittal", "coronal", "axial")}


def _value(v):
    if isinstance(v, tuple):
        return [_value(x) for x in v]
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v


def plan(store) -> dict:
    """What the view is made of: ``{"path", "overlays", "commands",
    "mode", "plane", "planes", "skipped"}``. ``commands`` are ``(id,
    params)`` pairs, with overlay layers renamed to the ids a fresh open
    gives them (``overlay1``, ``overlay2``, ... in order)."""
    scene = store.scene
    base = scene.base_layer()
    if base is None:
        raise ValueError("no image is open")
    ref = scene.sources.get(base.source)
    path = ref.path if ref is not None else ""
    overlays: list[str] = []
    skipped: list[str] = []
    ids = {base.id: "base"}
    for layer in scene.layers:
        if layer.kind != "volume" or layer.id == base.id:
            continue
        src_ref = scene.sources.get(layer.source)
        if src_ref is None or not src_ref.path:
            skipped.append(layer.name or layer.id)
            continue
        overlays.append(src_ref.path)
        ids[layer.id] = f"overlay{len(overlays)}"
    default = VolumeDisplay()
    commands: list[tuple[str, dict]] = []
    for layer in scene.layers:
        new_id = ids.get(layer.id)
        if new_id is None:
            continue
        params: dict = {}
        for name in DISPLAY_FIELDS:
            value = getattr(layer.display, name)
            if value is not None and value != getattr(default, name):
                params[name] = _value(value)
            elif new_id != "base" and value is not None:
                # An overlay opens in the look its contents call for, not the
                # default: say every field, so the look is this one.
                params[name] = _value(value)
        if not layer.visible:
            params["visible"] = False
        if params:
            commands.append(("layer.set", {"layer": new_id, **params}))
        if getattr(layer, "frame", 0):
            commands.append(("frame.set", {"frame": int(layer.frame), "layer": new_id}))
    world = scene.cursor.world
    if world is not None:
        x, y, z = (float(f"{float(v):.4f}") for v in world)
        commands.append(("cursor.set_world", {"x": x, "y": y, "z": z, "snap": False}))
    mode = scene.mode
    planes = LAYOUT_PLANES.get(mode, (scene.plane,))
    return {"path": path, "overlays": overlays, "commands": commands, "mode": mode,
            "plane": scene.plane, "planes": planes, "skipped": skipped}


def _param_text(key: str, value) -> str:
    if isinstance(value, str):
        return f"{key}={value}"
    return f"{key}={json.dumps(value, separators=(',', ':'))}"


#: An argument Windows shells take bare. Anything else is double-quoted: in
#: PowerShell a bare comma makes an array and brackets are special.
_WINDOWS_BARE = re.compile(r"[A-Za-z0-9_\-./:=\\]+")


def _quote(arg: str, style: str) -> str:
    if style == "posix":
        return shlex.quote(arg)
    if _WINDOWS_BARE.fullmatch(arg):
        return arg
    return '"' + arg.replace('"', '\\"') + '"'


def shell(store, *, render: Optional[str] = None, program: str = "bidsmgr-view",
          style: Optional[Literal["posix", "windows"]] = None) -> str:
    """A ``bidsmgr-view`` call that opens this view; with ``render`` set to
    a file name, one that writes its 2-D planes to that PNG with no window.

    ``style`` is the shell's: POSIX quoting and backslash continuations for
    macOS and Linux, double quotes on ONE line for cmd and PowerShell (which
    read neither). Default: this machine's."""
    style = style or ("windows" if os.name == "nt" else "posix")
    q = lambda arg: _quote(str(arg), style)  # noqa: E731
    p = plan(store)
    lines = [f"{program} {q(p['path'])}"]
    for extra in p["overlays"]:
        lines.append(f"--overlay {q(extra)}")
    if render is None:
        lines.append(f"--layout {p['mode']} --plane {p['plane']}")
    for command_id, params in p["commands"]:
        args = " ".join(q(_param_text(k, v)) for k, v in params.items())
        lines.append(f"--run {command_id} {args}".rstrip())
    if render is not None:
        lines.append(f"--planes {' '.join(p['planes'])} --render {q(render)}")
    if style != "posix":
        # cmd has no "#" comment (and PowerShell no REM): what was left out
        # is said by the caller, not in the command.
        return " ".join(lines)
    return _with_skipped(" \\\n    ".join(lines), p["skipped"])


def python(store, *, out: str = "figure.png") -> str:
    """The same view as a script over :mod:`bidsmgr.viz.render2d`."""
    p = plan(store)
    overlays = ", ".join(repr(str(o)) for o in p["overlays"])
    lines = ["from bidsmgr.viz import render2d", "",
             f"store = render2d.open_store({str(p['path'])!r}"
             + (f", overlays=[{overlays}])" if overlays else ")")]
    for command_id, params in p["commands"]:
        args = ", ".join(f"{k}={_py(v)}" for k, v in params.items())
        lines.append(f"store.run({command_id!r}, {args})")
    planes = ", ".join(repr(x) for x in p["planes"])
    lines += [f"rgba = render2d.render(store, planes=({planes},))",
              f"render2d.write_png({out!r}, rgba)"]
    return _with_skipped("\n".join(lines), p["skipped"])


def _py(value) -> str:
    if isinstance(value, list):
        return "(" + ", ".join(_py(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


def _with_skipped(text: str, skipped: list[str]) -> str:
    if not skipped:
        return text
    names = ", ".join(skipped)
    return (f"# Not reproduced: {names} (computed in the viewer, there is no file "
            "to open)\n" + text)


def parse_shell(text: str) -> list[str]:
    """The arguments of a :func:`shell` text, comments and continuations
    removed (tests, and a host that runs it in-process)."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return shlex.split(body.replace("\\\n", " "))[1:]


__all__ = ["DISPLAY_FIELDS", "parse_shell", "plan", "python", "shell"]

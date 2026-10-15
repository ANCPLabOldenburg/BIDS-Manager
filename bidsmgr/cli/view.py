"""``bidsmgr-view``: the viewer on its own, with no Editor around it.

Usage::

    bidsmgr-view PATH [--root DIR] [--overlay IMAGE ...] [--scene NAME]
                 [--layout MODE] [--plane PLANE]
                 [--colormap NAME] [--window LO HI] [--frame N]
                 [--run COMMAND [KEY=VALUE ...]] [--screenshot OUT.png]
                 [--scale S] [--transparent] [--size W H] [--theme THEME]
    bidsmgr-view PATH [--overlay IMAGE ...] [...] --render OUT.png
                 [--planes PLANE ...] [--height PX] [--crosshair]

Opens one image the way the Editor's viewer would, through the same library
(``bidsmgr.gui.viz.Viewer``). It exists for two reasons. It is the proof that
the library's boundary is clean: if the viewer can stand without the Editor,
nothing in it reaches back into the Editor. And it is useful on its own: a
quick look at a file from a terminal, or a scripted figure.

``--screenshot`` renders the view and writes a PNG without waiting for a
person, which with ``--run`` makes a batch QC figure from a shell loop::

    for f in sub-*/anat/*_T1w.nii.gz; do
        bidsmgr-view "$f" --layout mosaic --run view.mosaic_text "text=A -20 0 20 40" \\
                     --screenshot "qc/$(basename "$f" .nii.gz).png"
    done

``--overlay`` draws more images over it (an atlas, a statistical map, an MRS
file as its voxel), each in the look its contents call for. ``--scene`` opens
a scene saved in the dataset (Views menu), PATH then naming the dataset or
any file in it::

    bidsmgr-view /data/study --scene "hippocampus check" --screenshot fig.png

``--render`` writes the 2-D planes to a PNG with NO window, no Qt and no
display (``bidsmgr.viz.render2d``): on a server, in a container, in a batch
job. It draws exactly what the viewer's slices draw::

    bidsmgr-view sub-01_T1w.nii.gz --overlay sub-01_dseg.nii.gz \\
                 --planes sagittal coronal axial --height 320 --render qc.png

The viewer's View menu shows the command line of what is on screen (the
files, their looks, the crosshair), in this form and as Python.

The dataset root (for the sidecar, the events, the relative path in the
footer) is found by walking up to ``dataset_description.json`` unless given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

LAYOUTS = ("single", "multi", "3d", "combo", "hero", "mosaic")
PLANES = ("axial", "coronal", "sagittal")


def find_root(path: Path) -> Optional[Path]:
    """The nearest folder above ``path`` holding ``dataset_description.json``."""
    for parent in [path.parent, *path.parents]:
        if (parent / "dataset_description.json").is_file():
            return parent
    return None


def _parse_params(items: list[str]) -> dict:
    """``key=value`` pairs; values are read as JSON when they parse
    (``n=3``, ``window=[0,100]``, ``value=true``), else kept as text."""
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            params[key] = json.loads(raw)
        except ValueError:
            params[key] = raw
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-view",
        description="Open an image in the BIDS Manager viewer.",
    )
    parser.add_argument("path", type=Path,
                        help="The image to open (.nii, .nii.gz, .mgz, ...); with --scene, "
                             "the dataset or any file in it.")
    parser.add_argument("--root", type=Path, default=None,
                        help="The dataset root (default: found from the path).")
    parser.add_argument("--overlay", type=Path, action="append", default=[],
                        metavar="IMAGE", help="Draw this image over it (repeatable).")
    parser.add_argument("--scene", default=None, metavar="NAME",
                        help="Open a scene saved in the dataset (its name, or its .json).")
    parser.add_argument("--layout", choices=LAYOUTS, default=None,
                        help="How to show it (default: the layout you last used).")
    parser.add_argument("--plane", choices=PLANES, default=None,
                        help="The plane of the single and hero layouts.")
    parser.add_argument("--colormap", default=None, help="Colour map name, e.g. hot, viridis.")
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None,
                        help="Window in the data's own units.")
    parser.add_argument("--frame", type=int, default=None, help="Volume of a 4-D series (from 0).")
    parser.add_argument("--run", nargs="+", action="append", default=[],
                        metavar=("COMMAND", "KEY=VALUE"),
                        help="Run a viewer command after opening (repeatable), "
                             "e.g. --run cursor.set_voxel i=10 j=20 k=30")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="Write the view to this PNG and exit.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Screenshot scale: 2 draws the figure at twice the "
                             "window's pixels, sharp, for print (2-D views).")
    parser.add_argument("--transparent", action="store_true",
                        help="Screenshot with a transparent background instead "
                             "of the black surround.")
    parser.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=(1100, 760),
                        help="Window size in pixels (default 1100 760).")
    # Checked when the window opens, not here: the theme list lives in the
    # GUI package, and --render must parse and run with no Qt at all.
    parser.add_argument("--theme", default=None, metavar="THEME",
                        help="The colour theme: dark, dim, nord, hc-dark, light, paper or "
                             "hc-light (default: the one last chosen in the app).")
    parser.add_argument("--render", type=Path, default=None, metavar="OUT.png",
                        help="Write the 2-D planes to this PNG with no window, no Qt and "
                             "no display, and exit.")
    parser.add_argument("--planes", nargs="+", choices=PLANES, default=None,
                        help="With --render: the planes, side by side "
                             "(default: --plane, else axial).")
    parser.add_argument("--height", type=int, default=None, metavar="PX",
                        help="With --render: the height of every plane in pixels "
                             "(default: 2 pixels per millimetre, times --scale).")
    parser.add_argument("--crosshair", action="store_true",
                        help="With --render: draw the crosshair.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    path = args.path.expanduser()
    if not path.exists():
        print(f"bidsmgr-view: no such file: {path}", file=sys.stderr)
        return 2
    try:
        commands = [(cmd[0], _parse_params(cmd[1:])) for cmd in args.run]
    except ValueError as exc:
        print(f"bidsmgr-view: {exc}", file=sys.stderr)
        return 2
    root = args.root or find_root(path.resolve() if path.is_file() else path.resolve() / "x")
    scene_file = None
    if args.scene:
        from ..viz import scenes

        candidate = Path(args.scene).expanduser()
        if candidate.suffix == ".json" and candidate.is_file():
            scene_file = candidate
        elif root is not None:
            scene_file = scenes.scene_path(root, args.scene)
        if scene_file is None or not scene_file.is_file():
            print(f"bidsmgr-view: no scene {args.scene!r} in {root or path}", file=sys.stderr)
            return 2
        try:
            expected_overlays = len(scenes.load(scene_file).get("overlays", []))
        except (OSError, ValueError) as exc:
            print(f"bidsmgr-view: {exc}", file=sys.stderr)
            return 2
    else:
        expected_overlays = len(args.overlay)
    for extra in args.overlay:
        if not extra.expanduser().exists():
            print(f"bidsmgr-view: no such file: {extra}", file=sys.stderr)
            return 2
    if args.render is not None:
        if scene_file is not None:
            print("bidsmgr-view: --render draws files, not saved scenes; use "
                  "--screenshot for a scene", file=sys.stderr)
            return 2
        return _render(args, path, commands)

    from ..gui.bootstrap import create_application
    from ..gui.theme_manager import theme_ids

    if args.theme is not None and args.theme not in theme_ids():
        print(f"bidsmgr-view: no theme {args.theme!r}; choose from "
              f"{', '.join(theme_ids())}", file=sys.stderr)
        return 2
    app, _theme = create_application(args.theme)

    from ..gui.viz import Viewer
    from ..viz.commands import COMMANDS

    for name, _params in commands:
        if name not in COMMANDS:
            print(f"bidsmgr-view: unknown command {name!r}", file=sys.stderr)
            return 2

    from ..gui.viz.bridge import SettingsHub

    # A screenshot is the figure at the size asked for: no controls column.
    viewer = Viewer(kind="volume", panels=args.screenshot is None)
    viewer.setWindowTitle(f"{path.name} - BIDS Manager")
    viewer.resize(*args.size)
    outcome = {"code": 0}
    hub = SettingsHub.instance()

    def apply_options() -> None:
        if args.layout:
            viewer.run("view.mode", mode=args.layout)
        if args.plane:
            viewer.run("view.plane", plane=args.plane)
        if args.colormap:
            viewer.run("layer.set", colormap=args.colormap)
        if args.window:
            viewer.run("layer.set", window=tuple(args.window))
        if args.frame is not None:
            viewer.run("frame.set", frame=args.frame)
        for name, params in commands:
            viewer.run(name, **params)
        viewer.qstore.flush()

    def shoot() -> None:
        viewer.save_screenshot(args.screenshot, scale=args.scale,
                               transparent=args.transparent)
        print(args.screenshot)
        app.quit()

    state = {"loaded": False, "overlays": 0, "shot": False}

    def maybe_shoot() -> None:
        """Once the image AND every overlay are on screen."""
        if (args.screenshot is not None and state["loaded"] and not state["shot"]
                and state["overlays"] >= expected_overlays):
            state["shot"] = True
            # A short wait: one turn to lay the new layout out, one to paint
            # it, then the grab.
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(50, shoot)

    def on_overlay(_layer_id: str) -> None:
        state["overlays"] += 1
        maybe_shoot()

    def on_loaded(_p) -> None:
        if state["loaded"]:
            return
        state["loaded"] = True
        for extra in args.overlay:
            viewer.add_overlay(extra.expanduser())
        try:
            # The options are a one-off: applied, never remembered as the
            # user's preference (a batch of QC figures must not change the
            # layout the GUI opens in).
            with hub.suspended():
                apply_options()
        except Exception as exc:  # noqa: BLE001 - a bad argument ends the run, readably
            print(f"bidsmgr-view: {exc}", file=sys.stderr)
            outcome["code"] = 2
            app.quit()
            return
        maybe_shoot()

    def on_failed(_p, message: str) -> None:
        print(f"bidsmgr-view: could not open {path.name}: {message}", file=sys.stderr)
        outcome["code"] = 1
        if args.screenshot is not None:
            app.quit()

    viewer.loaded.connect(on_loaded)
    viewer.load_failed.connect(on_failed)
    viewer.overlay_added.connect(on_overlay)
    viewer.show()
    if scene_file is not None:
        if not viewer.presenter.open_scene(scene_file):
            print(f"bidsmgr-view: could not open the scene {args.scene!r}", file=sys.stderr)
            return 1
    else:
        viewer.set_file(path, root)
    code = app.exec()
    viewer.stop_loading()
    return outcome["code"] or code


def _render(args, path: Path, commands: list[tuple[str, dict]]) -> int:
    """``--render``: the figure with no window (``viz.render2d``). Qt is
    never imported, so this runs where there is no display at all."""
    from ..viz import render2d
    from ..viz.commands import COMMANDS

    for name, _params in commands:
        if name not in COMMANDS:
            print(f"bidsmgr-view: unknown command {name!r}", file=sys.stderr)
            return 2
    steps: list[tuple[str, dict]] = []
    if args.colormap:
        steps.append(("layer.set", {"colormap": args.colormap}))
    if args.window:
        steps.append(("layer.set", {"window": tuple(args.window)}))
    if args.frame is not None:
        steps.append(("frame.set", {"frame": args.frame}))
    steps += commands
    planes = tuple(args.planes or ((args.plane,) if args.plane else ("axial",)))
    try:
        out = render2d.render_file(
            path, args.render.expanduser(), overlays=[o.expanduser() for o in args.overlay],
            planes=planes, commands=steps, height_px=args.height,
            px_per_mm=2.0 * args.scale, crosshair=args.crosshair,
            transparent=args.transparent)
    except Exception as exc:  # noqa: BLE001 - a bad argument ends the run, readably
        print(f"bidsmgr-view: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

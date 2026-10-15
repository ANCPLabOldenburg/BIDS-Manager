"""The visualisation library: one engine for volumes, signals and spectra.

Qt-free. The layers, bottom to top:

* :mod:`bidsmgr.viz.data`: sources, what a file is and how to read it;
* :mod:`bidsmgr.viz.scene`: what is shown, as plain data;
* :mod:`bidsmgr.viz.commands` and :mod:`bidsmgr.viz.store`: the only way the
  scene changes, with undo and change notification;
* :mod:`bidsmgr.viz.compute`: the maths, one implementation each;
* :mod:`bidsmgr.viz.render2d`: a scene's slices as pixels or a PNG, with no
  window; :mod:`bidsmgr.viz.reproduce`: the command line of a view;
* :mod:`bidsmgr.viz.actions`, :mod:`bidsmgr.viz.inputmap`,
  :mod:`bidsmgr.viz.settings`, :mod:`bidsmgr.viz.theme`: what the user can
  configure, as data.

Using it from another tool
--------------------------

A figure, with no Qt and no display (the same pixels the viewer draws)::

    from bidsmgr.viz import render2d

    store = render2d.open_store("sub-01_T1w.nii.gz", overlays=["sub-01_dseg.nii.gz"])
    store.run("layer.set", layer="base", colormap="gray", window=(0, 900))
    store.run("cursor.set_world", x=0.0, y=-18.0, z=12.0)
    rgba = render2d.render(store, planes=("sagittal", "coronal", "axial"), height_px=320)
    render2d.write_png("sub-01.png", rgba)

The shell form is ``bidsmgr-view PATH ... --render OUT.png``. Every command
(``bidsmgr.viz.commands.COMMANDS``, each with its title and validated
parameters) works on a store whether or not a window shows it; the viewer's
toolbar, keys and menus only run them.

The interactive viewer, embedded in any PyQt6 window, is
:class:`bidsmgr.gui.viz.Viewer` (see :mod:`bidsmgr.gui.viz`).

The design and its reasons: workspace
``ancp_development_context_mds/active/VISUALIZATION_LIBRARY_PLAN.md``.
"""

from .scene import Scene
from .store import SceneStore

__all__ = ["Scene", "SceneStore"]

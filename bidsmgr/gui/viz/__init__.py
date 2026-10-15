"""The Qt half of the visualisation library: canvases, viewer, links.

``Viewer`` is the widget a host embeds; everything it shows is a
:class:`bidsmgr.viz.scene.Scene` changed through commands. See
:mod:`bidsmgr.viz` for the Qt-free half, including rendering with no window.

Embedding it
------------

::

    from bidsmgr.gui.viz import Viewer, link

    viewer = Viewer(kind="volume")          # or "signal" (MEG, EEG, physio),
    layout.addWidget(viewer)                #    "spectrum" (MR spectroscopy)
    viewer.loaded.connect(on_ready)         # the whole file is read
    viewer.set_file(path, dataset_root)     # read on a worker thread

    def on_ready(_path):
        viewer.add_overlay(atlas_path)      # drawn in the look it calls for
        viewer.run("layer.set", colormap="viridis", window=(0, 900))
        viewer.run("view.mode", mode="multi")
        viewer.trigger("view.inspector")    # any action, as its key would
        viewer.save_screenshot(Path("figure.png"), scale=2)

    link(viewer, other_viewer, cursor=True, view=True)   # two in step

Signals: ``loaded``, ``load_failed``, ``overlay_added``, ``status_message``
(a line for a status bar), ``loading_changed`` (a host's busy spinner),
``close_requested``. ``Viewer(panels=False)`` keeps the controls column
shut, for a host that wants the figure alone. Whatever is on screen can be
saved and re-applied as plain data (``viewer.state()``,
``viewer.apply_state(...)``), and "Show the command line" in the Views menu
writes the call that reproduces it.
"""

from .link import ViewerLink, link
from .viewer import Viewer

__all__ = ["Viewer", "ViewerLink", "link"]

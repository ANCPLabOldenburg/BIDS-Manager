"""What every canvas and panel of one viewer shares.

One :class:`ViewerContext` per viewer: its store, its job runner, the
process-wide theme and settings hubs, and the small amount of state that is
the GUI's alone (which view the pointer last used, whether a key is held,
whether frames are playing). Canvases receive it instead of a pointer to the
viewer, so a canvas can be placed in any layout, or tested alone.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .bridge import JobRunner, QtStore, SettingsHub, ThemeHub


class ViewerContext(QObject):
    """Shared state of one viewer's canvases."""

    #: The plane the pointer last interacted with changed.
    active_plane_changed = pyqtSignal(str)
    #: Play state changed (frames advancing on a timer).
    playing_changed = pyqtSignal(bool)
    #: A canvas asks the host to show a message in the status bar.
    status = pyqtSignal(str)
    #: A canvas has a new position readout (the value under the pointer).
    readout = pyqtSignal(str)
    #: The 3-D panel chose another clip plane to edit.
    active_clip_changed = pyqtSignal(int)
    #: The controls column selected another layer (its id).
    layer_selected = pyqtSignal(str)

    def __init__(self, qstore: QtStore, jobs: JobRunner, parent=None) -> None:
        super().__init__(parent)
        self.qstore = qstore
        self.jobs = jobs
        self.theme_hub = ThemeHub.instance()
        self.settings_hub = SettingsHub.instance()
        self.active_plane: str = "axial"
        #: Keys held down while the viewer has focus ("h" for the frame wheel).
        self.held: set[str] = set()
        self.playing = False
        self.gpu_ok = False
        #: The clip plane the 3-D panel is editing; gestures and the Shift
        #: keys act on it.
        self.active_clip = 0
        #: The layer the controls column edits ("" = the base image).
        self.selected_layer = ""
        #: True while a figure with a transparent background is being drawn:
        #: canvases leave out their surround.
        self.transparent = False
        #: Set by the viewer: run an action id (canvases' context menus).
        self.run_action: Optional[Callable[[str], None]] = None

    @property
    def store(self):
        return self.qstore.store

    @property
    def scene(self):
        return self.qstore.store.scene

    @property
    def settings(self):
        return self.settings_hub.settings

    @property
    def theme(self):
        return self.theme_hub.theme

    def run(self, command_id: str, /, **params):
        return self.qstore.run(command_id, **params)

    def select_layer(self, layer_id: str) -> None:
        if layer_id != self.selected_layer:
            self.selected_layer = layer_id
            self.layer_selected.emit(layer_id)

    def set_active_plane(self, plane: str) -> None:
        if plane != self.active_plane:
            self.active_plane = plane
            self.active_plane_changed.emit(plane)


__all__ = ["ViewerContext"]

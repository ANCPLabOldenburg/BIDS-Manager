"""Background loader for NIfTI volumes.

The Editor's :class:`bidsmgr.gui.widgets.NiftiViewerPane` loads big
fMRI / dMRI 4-D arrays — a typical BOLD run can hit several hundred
MB, and ``nibabel.Nifti1Image.get_fdata()`` blocks while the file
is read + decompressed. Doing that on the GUI thread freezes the
window, including the spinner that's supposed to communicate
"loading".

This worker pushes that load onto a ``QThread`` and emits signals
the pane connects to.

Cancellation
------------
The pane often issues a new load before the previous one has
finished (user clicks a different file in the BIDS tree). The
worker doesn't try to *interrupt* the in-flight ``get_fdata`` — it
just flips a flag and suppresses the result emission when it
finally arrives, so the now-stale data never reaches the GUI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


log = logging.getLogger(__name__)


class NiftiLoaderWorker(QThread):
    """Loads a NIfTI on a background thread.

    Signals
    -------
    finished_with_data(object, object, dict, Path)
        Emitted on successful load. Args: ``(nibabel_img, data_array,
        meta_dict, path)``. Always check ``path`` against the
        currently-bound file before consuming.
    failed(Path, str)
        Emitted when load fails. Args: ``(path, error_message)``.
    """

    finished_with_data = pyqtSignal(object, object, dict, Path)
    failed = pyqtSignal(Path, str)

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._cancelled = False

    def path(self) -> Path:
        return self._path

    def cancel(self) -> None:
        """Suppress this worker's result emissions.

        The actual NIfTI read can't be interrupted mid-flight, but
        once it returns we silently drop the result instead of
        emitting it.
        """
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        # Local import keeps the Qt import graph free of nibabel.
        from bidsmgr.gui.widgets.nifti_viewer_pane import _load_nifti

        try:
            img, data, meta = _load_nifti(self._path)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            log.warning("NiftiLoaderWorker failed on %s: %s", self._path, exc)
            if not self._cancelled:
                self.failed.emit(self._path, str(exc))
            return

        if self._cancelled:
            return

        # ``np.asarray`` materialises the data on this thread so the
        # GUI doesn't pay the cost when accessing it later. mmap-backed
        # arrays in particular load lazily otherwise.
        try:
            data = np.asarray(data)
        except Exception:  # pragma: no cover - defensive
            pass

        self.finished_with_data.emit(img, data, meta, self._path)


__all__ = ["NiftiLoaderWorker"]

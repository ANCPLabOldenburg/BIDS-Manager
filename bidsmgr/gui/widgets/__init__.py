"""Reusable GUI widgets.

Public surface — every widget the Converter and Editor views compose:

* :class:`Chip`, :class:`VSep`, :class:`PaneHeader`, :class:`PathBar`
  — generic primitives (toolbar chips, separators, pane headers,
  path-display strips).
* :class:`StatusBadge` + :func:`badge_paint` — severity badge widget
  and the paint helper the table delegates re-use.
* :class:`SidecarRow`     — one row of the Editor's sidecar form.
* :class:`ValMessage`     — one validator finding rendered as a row.

Lift-and-shift from ``inspector_proto/proto.py``; no logic change.
"""

from .bids_tree_pane import BidsTreePane
from .image_label import ImageLabel
from .json_tree_view import JsonTreeView
from .nifti_viewer_pane import NiftiViewerPane
from .primitives import Chip, PaneHeader, PathBar, VSep
from .sidecar_form_pane import SidecarFormPane, find_peer_files
from .sidecar_row import SidecarRow
from .tsv_viewer_pane import TsvViewerPane
from .validation_pane import ValidationPane
from .spinner import BusySpinner
from .status_badge import (
    KIND_BG_TOKEN,
    KIND_CHAR,
    KIND_FG_TOKEN,
    StatusBadge,
    badge_paint,
)
from .val_message import ValMessage

__all__ = [
    "BidsTreePane",
    "BusySpinner",
    "Chip",
    "ImageLabel",
    "JsonTreeView",
    "KIND_BG_TOKEN",
    "KIND_CHAR",
    "KIND_FG_TOKEN",
    "NiftiViewerPane",
    "PaneHeader",
    "PathBar",
    "SidecarFormPane",
    "SidecarRow",
    "TsvViewerPane",
    "ValidationPane",
    "find_peer_files",
    "StatusBadge",
    "ValMessage",
    "VSep",
    "badge_paint",
]

"""QStyledItemDelegate paint classes used by Converter + Editor views.

Public surface:

* Converter inspection table:
    - :class:`StatusDelegate`    — status-badge column.
    - :class:`CheckboxDelegate`  — include-toggle column.
    - :class:`CellTextDelegate`  — every text column (with role-based
      formatting: ``plain``, ``mono``, ``basename``, ``conf``).
* Editor BIDS tree:
    - :class:`BidsTreeDelegate`  — paints per-row validation badges.

Shared helpers:

* :func:`paint_row_state` + :data:`ROW_STATE_ROLE` — every inspection
  delegate calls this first so the row's selection / severity tint is
  applied beneath whatever the cell draws.
* :data:`PAYLOAD_ROLE`, :data:`BADGE_ROLE` — role indices the models
  use to publish badge kinds and tree-row severities.
"""

from .bids_tree import BADGE_ROLE, BidsTreeDelegate
from .inspection_cells import (
    PAYLOAD_ROLE,
    CellTextDelegate,
    CheckboxDelegate,
    StatusDelegate,
)
from .row_state import (
    HIGHLIGHT_ROLE,
    ROW_STATE_ROLE,
    paint_highlight,
    paint_row_state,
)

__all__ = [
    "BADGE_ROLE",
    "BidsTreeDelegate",
    "CellTextDelegate",
    "CheckboxDelegate",
    "HIGHLIGHT_ROLE",
    "PAYLOAD_ROLE",
    "ROW_STATE_ROLE",
    "StatusDelegate",
    "paint_highlight",
    "paint_row_state",
]

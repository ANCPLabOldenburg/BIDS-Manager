"""QAbstractItemModel subclasses bound to ``bidsmgr`` data.

* :class:`InventoryTableModel` — the Converter view's table model. Wraps
  the unified-TSV DataFrame and overlays a :class:`bidsmgr.project.Project`
  if one is attached.
* :data:`COLUMNS` / :class:`ColumnSpec` — the 12 display columns the
  model exposes, in display order. Views read this to set per-column
  widths + delegate roles.
"""

from .inventory import (
    COLUMNS,
    MANDATORY_COLUMN_KEYS,
    ColumnSpec,
    InventoryTableModel,
)

__all__ = [
    "COLUMNS",
    "ColumnSpec",
    "InventoryTableModel",
    "MANDATORY_COLUMN_KEYS",
]

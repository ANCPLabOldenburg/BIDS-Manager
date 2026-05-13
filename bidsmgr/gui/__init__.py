"""The only Qt-coupled subtree.

Reference: architecture.md §8 + the visual prototype at
``../inspector_proto/`` (chosen GUI shape: Inspector,
super_plan.md §13).

Architectural rule (architecture.md §12):
**Nothing in the rest of the package imports from here.** ``gui``
imports core modules; the dependency arrow only points inward.

Submodules:
* ``theme_manager``    — palette swap at runtime (token-based QSS).
* ``main_window``      — top header (view switcher + theme toggle)
  + ``QStackedWidget`` (Converter + Editor).
* ``converter_panel``  — the 4-column Inspector layout (raw FS
  tree, modality filter w/ tri-state checkboxes, inspection
  QTableView, properties panel).
* ``editor_panel``     — 3-column post-conv editor (BIDS tree
  with validation badges, type-routed viewer, validation panel).
* ``widgets/``         — reusable: chip, status badge, path bar,
  inventory table, entity form, sidecar form, modality tree.
* ``delegates/``       — QStyledItemDelegate subclasses for the
  inventory table and BIDS tree.
* ``models/``          — QAbstractTableModel for the inventory.

``theme.qss`` is the token-based stylesheet (loaded by
``theme_manager``). Seed copy lives in this package; the
prototype's source lives at ``../../inspector_proto/theme.qss``
for visual comparison.
"""

from .converter_panel import ConverterPanel
from .theme_manager import DARK, LIGHT, PALETTES, ThemeManager

__all__ = ["ConverterPanel", "DARK", "LIGHT", "PALETTES", "ThemeManager"]

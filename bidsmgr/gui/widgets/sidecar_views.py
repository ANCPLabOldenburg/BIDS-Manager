"""The pieces of the Sidecars from the recordings dialog.

- :class:`RecordingTree`: the recordings whose sidecar would be written or
  completed, grouped by the folder they are in, each ticked, with what would
  happen to its sidecar.
- :class:`FieldList`: one sidecar's fields: what it gains and where each value
  was read, what a sidecar higher up already states, and what nothing could
  give. The origin is a painted pill, so a long sidecar is one view.

Qt only; the data are the plan's (:mod:`bidsmgr.editor.sidecars`).
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Optional

from PyQt6.QtCore import QModelIndex, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from ...editor.sidecars import (
    ANONYMISED,
    ANSWERS,
    FILENAME,
    HEADER,
    LINKS,
    ORIGIN_HELP,
    ORIGINS,
    PLACEHOLDER,
    SOURCE,
)
from .primitives import Chip, qcolor
from .tree_click import toggle_on_click

#: The plan of one recording, on its row of the tree.
PLAN_ROLE = Qt.ItemDataRole.UserRole + 1
#: A pill's colour set (a :class:`Chip` kind), on the field list's last column.
TONE_ROLE = Qt.ItemDataRole.UserRole + 2

#: The colour of each origin. One accent is for actions (4.19), so the origins
#: are told apart by the quieter sets; a placeholder is the one to act on.
ORIGIN_TONES: dict[str, str] = {
    SOURCE: "teal",
    HEADER: "default",
    ANSWERS: "purple",
    LINKS: "default",
    FILENAME: "default",
    ANONYMISED: "default",
    PLACEHOLDER: "warn",
}


def _value_text(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 400 else text[:399] + "…"


def short_name(relpath: str) -> str:
    """A recording's name without the subject and session its folder already
    names: ``sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz`` reads
    ``task-rest_bold.nii.gz`` under the folder ``sub-01/ses-1/func``."""
    path = PurePosixPath(relpath)
    named = {part for part in path.parts[:-1] if "-" in part}
    tokens = path.name.split("_")
    while len(tokens) > 1 and tokens[0] in named:
        tokens.pop(0)
    return "_".join(tokens)


def summary(plan) -> str:
    """What happens to one recording's sidecar, in a few words: the count of
    fields under the column "Fields", short enough to leave the name room."""
    n = len(plan.added)
    if not n:
        return "none to add"
    return f"{n} added" if plan.exists else f"{n}, new"


# ---------------------------------------------------------------------------
# The recordings
# ---------------------------------------------------------------------------


class RecordingTree(QTreeWidget):
    """The recordings, by folder, ticked when their sidecar would change."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("check-tree")
        self.setColumnCount(2)
        self.setHeaderLabels(["Recording", "Fields"])
        self.headerItem().setToolTip(
            1, "How many fields its sidecar gains: a new sidecar, or one added to.")
        self.setUniformRowHeights(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # The middle goes, not the end: the end is the suffix and extension,
        # which is what tells two recordings of one folder apart.
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        toggle_on_click(self)
        self.itemChanged.connect(self._on_item_changed)
        self._syncing = False

    def fill(self, plans: list, *, unfold: bool = True) -> None:
        """Show ``plans``, grouped by folder; those that add nothing unticked
        and greyed, so the tick is a decision about something."""
        self._syncing = True
        self.blockSignals(True)
        self.clear()
        folders: dict[str, QTreeWidgetItem] = {}
        for plan in plans:
            folder = str(PurePosixPath(plan.recording).parent)
            head = folders.get(folder)
            if head is None:
                head = QTreeWidgetItem([folder, ""])
                head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                head.setToolTip(0, folder)
                head.setFirstColumnSpanned(False)
                self.addTopLevelItem(head)
                folders[folder] = head
            item = QTreeWidgetItem(head, [short_name(plan.recording), summary(plan)])
            item.setData(0, PLAN_ROLE, plan)
            item.setToolTip(0, plan.recording)
            kind = getattr(plan, "kind", None)
            if kind is not None and not kind.bids:
                # Not a BIDS name or place: said in italics, as the tree says
                # an ignored file, and in words on hover.
                font = item.font(0)
                font.setItalic(True)
                item.setFont(0, font)
                item.setToolTip(0, f"{plan.recording}\nNot a BIDS name or place: read as "
                                   f"{kind.datatype}/{kind.suffix} ({kind.why}).")
            if plan.changes:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setForeground(0, qcolor(_pal("muted")))
                item.setForeground(1, qcolor(_pal("muted")))
        for head in folders.values():
            if not self._boxed(head):
                head.setFlags(head.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                head.setForeground(0, qcolor(_pal("muted")))
            else:
                head.setCheckState(0, Qt.CheckState.Checked)
            head.setExpanded(unfold)
        self.blockSignals(False)
        self._syncing = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """A folder ticks every recording in it that would change (the rest
        have no box); a recording sets its folder's tick to what it holds.
        Both ways by hand: Qt propagates a tristate only on a click, not on a
        tick set from code."""
        if self._syncing or column != 0:
            return
        self._syncing = True
        try:
            if item.data(0, PLAN_ROLE) is None:
                state = item.checkState(0)
                if state != Qt.CheckState.PartiallyChecked:
                    for child in self._boxed(item):
                        child.setCheckState(0, state)
            elif item.parent() is not None:
                self._sync_folder(item.parent())
        finally:
            self._syncing = False

    @staticmethod
    def _boxed(head: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        return [head.child(i) for i in range(head.childCount())
                if head.child(i).flags() & Qt.ItemFlag.ItemIsUserCheckable]

    def _sync_folder(self, head: QTreeWidgetItem) -> None:
        states = {c.checkState(0) for c in self._boxed(head)}
        if not states:
            return
        head.setCheckState(0, states.pop() if len(states) == 1
                           else Qt.CheckState.PartiallyChecked)

    def recordings(self) -> list[QTreeWidgetItem]:
        out = []
        for i in range(self.topLevelItemCount()):
            head = self.topLevelItem(i)
            out.extend(head.child(j) for j in range(head.childCount()))
        return out

    def chosen(self) -> list:
        """The plans ticked, in order."""
        return [item.data(0, PLAN_ROLE) for item in self.recordings()
                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable
                and item.checkState(0) == Qt.CheckState.Checked]

    def set_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for i in range(self.topLevelItemCount()):
            head = self.topLevelItem(i)
            if head.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                head.setCheckState(0, state)

    def selected_plan(self):
        items = self.selectedItems()
        return items[0].data(0, PLAN_ROLE) if items else None

    def select_recording(self, relpath: str) -> bool:
        """Select the row of ``relpath``; False when it is not listed."""
        for item in self.recordings():
            if item.data(0, PLAN_ROLE).recording == relpath:
                self.setCurrentItem(item)
                item.setSelected(True)
                return True
        return False

    def select_first(self) -> None:
        """Select the first recording that changes (else the first)."""
        rows = self.recordings()
        pick = next((r for r in rows if r.data(0, PLAN_ROLE).changes), rows[0] if rows else None)
        if pick is not None:
            self.setCurrentItem(pick)
            pick.setSelected(True)


def _pal(token: str) -> str:
    from ..theme_manager import CUR

    return CUR().get(token, "#888888")


# ---------------------------------------------------------------------------
# One sidecar's fields
# ---------------------------------------------------------------------------


class _PillDelegate(QStyledItemDelegate):
    """The last column as a pill in its tone; everything else as usual."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        tone = index.data(TONE_ROLE)
        if not tone:
            super().paint(painter, option, index)
            return
        from ..viz import fonts

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        bg, fg, border = Chip._KIND_TOKENS.get(tone, Chip._KIND_TOKENS["default"])
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        small = fonts.font(10)
        metrics = QFontMetrics(small)
        width = min(metrics.horizontalAdvance(text) + fonts.px(14),
                    max(0, option.rect.width() - fonts.px(4)))
        height = metrics.height() + fonts.px(2)
        pill = QRectF(option.rect.left() + fonts.px(2),
                      option.rect.center().y() - height / 2.0, width, height)
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(qcolor(_pal(border)), 1))
            painter.setBrush(qcolor(_pal(bg)))
            painter.drawRoundedRect(pill, height / 2.0, height / 2.0)
            painter.setFont(small)
            painter.setPen(QColor(qcolor(_pal(fg))))
            painter.drawText(pill.toRect(), Qt.AlignmentFlag.AlignCenter,
                             metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                                int(pill.width()) - fonts.px(8)))
        finally:
            painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        hint = super().sizeHint(option, index)
        if index.data(TONE_ROLE):
            from ..viz import fonts

            metrics = QFontMetrics(fonts.font(10))
            text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            hint.setWidth(metrics.horizontalAdvance(text) + fonts.px(22))
        return hint


class FieldList(QTreeWidget):
    """One sidecar: the fields it gains, those left to a sidecar higher up,
    and the required ones nothing could give."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("check-tree")
        self.setColumnCount(3)
        self.setHeaderLabels(["Field", "Value", "Read from"])
        self.setUniformRowHeights(True)
        self.setRootIsDecorated(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setItemDelegate(_PillDelegate(self))
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        toggle_on_click(self)

    def show_plan(self, plan) -> None:
        from ..viz import fonts

        self.clear()
        if plan is None:
            return
        mono = fonts.font(11, mono=True)
        bold = fonts.font(11, bold=True)

        def heading(text: str, count: int) -> QTreeWidgetItem:
            item = QTreeWidgetItem([f"{text}  ·  {count}"])
            item.setFirstColumnSpanned(True)
            item.setFont(0, bold)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.addTopLevelItem(item)
            item.setExpanded(True)
            return item

        def row(parent, name: str, value: str, pill: str, tone: str, tip: str,
                muted: bool = False) -> None:
            item = QTreeWidgetItem(parent, [name, value, pill])
            item.setFont(0, mono)
            item.setFont(1, mono)
            item.setToolTip(0, name)
            item.setToolTip(1, value)
            item.setToolTip(2, tip)
            item.setData(2, TONE_ROLE, tone)
            if muted:
                item.setForeground(0, qcolor(_pal("muted")))
                item.setForeground(1, qcolor(_pal("muted")))

        if plan.added:
            head = heading("Written", len(plan.added))
            for f in plan.added:
                tip = ORIGIN_HELP.get(f.origin, "")
                if f.replaces_todo:
                    tip += "\n\nReplaces the TODO the sidecar held."
                if f.overrides:
                    tip += (f"\n\n{f.overrides} states {_value_text(f.overridden)} for "
                            "every recording it applies to. Written here, this value wins "
                            "for this recording only (the nearest sidecar wins).")
                row(head, f.name, _value_text(f.value), ORIGINS.get(f.origin, f.origin),
                    ORIGIN_TONES.get(f.origin, "default"), tip)
        if plan.inherited:
            head = heading("Already stated higher up, left to it", len(plan.inherited))
            for f in plan.inherited:
                tip = (f"{f.source} states this for every recording it applies to "
                       "(BIDS inheritance), so it is not repeated here.")
                if f.differs:
                    tip += (f"\n\nThis recording's own data say {_value_text(f.found)}: "
                            "check which is right.")
                row(head, f.name, _value_text(f.value),
                    "differs" if f.differs else PurePosixPath(f.source).name,
                    "warn" if f.differs else "default", tip, muted=not f.differs)
        if getattr(plan, "rejected", None):
            head = heading("Found, not written: not valid in this BIDS version",
                           len(plan.rejected))
            for r in plan.rejected:
                row(head, r.name, _value_text(r.value), "not valid", "err",
                    f"{r.problem}. Read from: {ORIGINS.get(r.origin, r.origin).lower()}. "
                    "Not written, so the sidecar stays valid; fill it in by hand if "
                    "the right value is known.", muted=True)
        if plan.still_missing:
            head = heading("Required, and nothing to read it from", len(plan.still_missing))
            for name in plan.still_missing:
                row(head, name, "", "missing", "err",
                    "The standard requires this field, and neither the recording, its "
                    "source data nor your answers hold it. Fill it in by hand.")
        self.resizeColumnToContents(0)
        # The names may take what they need, up to nearly half: the value is
        # what is being checked, and it must not be squeezed to a few letters.
        cap = max(fonts.px(120), self.viewport().width() * 9 // 20)
        self.setColumnWidth(0, min(self.columnWidth(0), cap))

    def visual_rect_of(self, name: str) -> Optional[QRect]:
        """Where the row of field ``name`` is drawn (for tests and captures)."""
        for i in range(self.topLevelItemCount()):
            head = self.topLevelItem(i)
            for j in range(head.childCount()):
                if head.child(j).text(0) == name:
                    return self.visualItemRect(head.child(j))
        return None


__all__ = ["FieldList", "ORIGIN_TONES", "PLAN_ROLE", "RecordingTree", "TONE_ROLE",
           "short_name", "summary"]

"""A list of findings that stays fast at any size.

The chip dialogs (the Editor's Errors / Warnings / Valid, the Converter's
Errors / Warnings / Skipped) built a card per file and about a dozen widgets
per finding, all at once. Measured on a 300-subject dataset: 59 s to open the
warnings of 300 files, more than 45 minutes for the whole report; the
Converter's 3,000-row warning list took 9 s. Every one of those seconds was
Qt styling and laying out widgets nobody had scrolled to.

This is one view over one model, painted (GUI_LOOK_AND_FEEL 8e: per-item
widgets do not scale; paint instead). Files (or inventory rows) are the
groups, their findings the children, one line each. The finding that is
selected is shown in full underneath, in the same :class:`ValMessage` the
validation pane uses, Fix button and all, because one widget costs nothing.
287,000 findings open in under a fifth of a second.

Qt-side only. The data is plain: :class:`Finding` and :class:`Group`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from PyQt6.QtCore import (
    QAbstractItemModel,
    QElapsedTimer,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class Finding:
    """One finding, as a line and as the full message underneath."""

    severity: str                 # err | warn | ok | skip
    message: str
    rule: str = ""
    field: Optional[str] = None
    fix_label: Optional[str] = None
    schema_rule: Optional[str] = None


@dataclass
class Group:
    """A file (or an inventory row) and its findings."""

    title: str
    key: Any                      # what activating it hands back: a Path, a row
    hint: str = ""                # beside the title: "func/bold"
    findings: list[Finding] = dc_field(default_factory=list)


KIND_ROLE = Qt.ItemDataRole.UserRole + 1
FINDING_ROLE = Qt.ItemDataRole.UserRole + 2
GROUP_ROLE = Qt.ItemDataRole.UserRole + 3

_TOP = 0xFFFFFFFF


class FindingsModel(QAbstractItemModel):
    """Groups and their findings, filtered by a search text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._groups: list[Group] = []
        self._shown: list[tuple[Group, list[Finding]]] = []
        self._query = ""

    # -- content --------------------------------------------------------

    def set_groups(self, groups: list[Group]) -> None:
        self.beginResetModel()
        self._groups = list(groups)
        self._shown = self._filter(self._query)
        self.endResetModel()

    def set_query(self, text: str) -> None:
        query = text.strip().lower()
        if query == self._query:
            return
        self.beginResetModel()
        self._query = query
        self._shown = self._filter(query)
        self.endResetModel()

    def _filter(self, query: str) -> list[tuple[Group, list[Finding]]]:
        """A group whose title matches keeps all its findings; otherwise only
        the findings that match, and only if there are any."""
        if not query:
            return [(g, g.findings) for g in self._groups]
        out = []
        for group in self._groups:
            if query in group.title.lower() or query in group.hint.lower():
                out.append((group, group.findings))
                continue
            kept = [f for f in group.findings
                    if query in f.message.lower() or query in f.rule.lower()
                    or query in (f.field or "").lower()]
            if kept:
                out.append((group, kept))
        return out

    def counts(self) -> tuple[int, int]:
        """``(groups, findings)`` shown."""
        return len(self._shown), sum(len(f) for _g, f in self._shown)

    def total(self) -> tuple[int, int]:
        return len(self._groups), sum(len(g.findings) for g in self._groups)

    # -- the tree -------------------------------------------------------

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if column != 0 or row < 0:
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, 0, _TOP) if row < len(self._shown) else QModelIndex()
        if parent.internalId() != _TOP:
            return QModelIndex()
        findings = self._shown[parent.row()][1]
        return self.createIndex(row, 0, parent.row()) if row < len(findings) else QModelIndex()

    def parent(self, child: QModelIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        if not child.isValid() or child.internalId() == _TOP:
            return QModelIndex()
        return self.createIndex(int(child.internalId()), 0, _TOP)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            return len(self._shown)
        if parent.internalId() == _TOP:
            return len(self._shown[parent.row()][1])
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 1

    def group_at(self, index: QModelIndex) -> Optional[Group]:
        if not index.isValid():
            return None
        row = index.row() if index.internalId() == _TOP else int(index.internalId())
        return self._shown[row][0] if 0 <= row < len(self._shown) else None

    def finding_at(self, index: QModelIndex) -> Optional[Finding]:
        if not index.isValid() or index.internalId() == _TOP:
            return None
        group, findings = self._shown[int(index.internalId())]
        return findings[index.row()] if index.row() < len(findings) else None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        top = index.internalId() == _TOP
        if role == KIND_ROLE:
            return "group" if top else "finding"
        if top:
            group, findings = self._shown[index.row()]
            if role == Qt.ItemDataRole.DisplayRole:
                return group.title
            if role == Qt.ItemDataRole.ToolTipRole:
                return group.title
            if role == GROUP_ROLE:
                return (group, len(findings))
            return None
        finding = self.finding_at(index)
        if finding is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return finding.message
        if role == Qt.ItemDataRole.ToolTipRole:
            return finding.message
        if role == FINDING_ROLE:
            return finding
        return None


def plain(html: str) -> str:
    """A finding's text on one line, without its markup."""
    import re
    from html import unescape

    text = re.sub(r"<[^>]+>", " ", html or "")
    return " ".join(unescape(text).split())


class FindingsDelegate(QStyledItemDelegate):
    """One painted line per row: a file, or a finding under it."""

    _COLOURS = {"err": "error", "warn": "warning", "ok": "success", "skip": "muted"}

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        from ..viz import fonts

        height = QFontMetrics(fonts.font(12)).height() + fonts.px(10)
        return QSize(option.rect.width(), height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        from ..theme_manager import CUR, rgba
        from ..viz import fonts

        pal = CUR()
        painter.save()
        try:
            rect = option.rect
            selected = bool(option.state & QStyle.StateFlag.State_Selected)
            hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
            if selected:
                painter.fillRect(rect, rgba(pal["accent"], 0.18))
            elif hovered:
                painter.fillRect(rect, rgba(pal["text"], 0.05))
            pad = fonts.px(6)
            inner = rect.adjusted(pad, 0, -pad, 0)
            if index.data(KIND_ROLE) == "group":
                self._paint_group(painter, inner, index, pal, fonts)
            else:
                self._paint_finding(painter, inner, index, pal, fonts)
        finally:
            painter.restore()

    def _paint_group(self, painter, rect: QRect, index, pal, fonts) -> None:
        group, count = index.data(GROUP_ROLE)
        small = fonts.font(11)
        painter.setFont(small)
        metrics = QFontMetrics(small)
        right = rect.right()
        count_text = f"{count}"
        pill_w = metrics.horizontalAdvance(count_text) + fonts.px(12)
        pill = QRect(right - pill_w, rect.center().y() - metrics.height() // 2 - 1,
                     pill_w, metrics.height() + 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal["surface3"]))
        painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        painter.setPen(QColor(pal["dim"]))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, count_text)
        right = pill.left() - fonts.px(8)
        if group.hint:
            hint_w = metrics.horizontalAdvance(group.hint)
            painter.setPen(QColor(pal["muted"]))
            painter.drawText(QRect(right - hint_w, rect.top(), hint_w, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             group.hint)
            right -= hint_w + fonts.px(10)
        title_font = fonts.font(12, bold=True)
        painter.setFont(title_font)
        painter.setPen(QColor(pal["text"]))
        title_rect = QRect(rect.left(), rect.top(), max(0, right - rect.left()), rect.height())
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         QFontMetrics(title_font).elidedText(
                             group.title, Qt.TextElideMode.ElideMiddle, title_rect.width()))

    def _paint_finding(self, painter, rect: QRect, index, pal, fonts) -> None:
        finding: Finding = index.data(FINDING_ROLE)
        dot = fonts.px(7)
        cy = rect.center().y()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal[self._COLOURS.get(finding.severity, "muted")]))
        painter.drawEllipse(rect.left(), cy - dot // 2, dot, dot)
        x = rect.left() + dot + fonts.px(8)
        label = finding.rule + (f" · {finding.field}" if finding.field else "")
        if label:
            mono = fonts.font(10, mono=True)
            painter.setFont(mono)
            painter.setPen(QColor(pal["dim"]))
            # One column for every row, so the messages line up.
            width = min(fonts.px(250), rect.width() * 2 // 5)
            painter.drawText(QRect(x, rect.top(), width, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             QFontMetrics(mono).elidedText(label, Qt.TextElideMode.ElideRight,
                                                           width))
            x += width + fonts.px(10)
        body = fonts.font(12)
        painter.setFont(body)
        painter.setPen(QColor(pal["text"]))
        text_rect = QRect(x, rect.top(), max(0, rect.right() - x), rect.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         QFontMetrics(body).elidedText(plain(finding.message),
                                                       Qt.TextElideMode.ElideRight,
                                                       text_rect.width()))


class FindingsView(QTreeView):
    """The tree of findings. A click on a file folds it (rule 4.31);
    double click or Return opens it."""

    activated_group = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("findings-view")
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.setRootIsDecorated(True)
        self.setMouseTracking(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setItemDelegate(FindingsDelegate(self))
        # Parented to the view (guard 8c): a view does not own its model.
        self.setModel(FindingsModel(self))
        self.setExpandsOnDoubleClick(False)
        self._last_toggle = (QModelIndex(), QElapsedTimer())
        self.clicked.connect(self._on_clicked)
        self.doubleClicked.connect(self._on_double_clicked)

    def findings_model(self) -> FindingsModel:
        return self.model()  # type: ignore[return-value]

    def _on_clicked(self, index: QModelIndex) -> None:
        if index.data(KIND_ROLE) != "group":
            return
        if QApplication.keyboardModifiers() != Qt.KeyboardModifier.NoModifier:
            return
        last, timer = self._last_toggle
        if last == index and timer.isValid() \
                and timer.elapsed() < QApplication.doubleClickInterval():
            return                 # the second half of a double click
        self.setExpanded(index, not self.isExpanded(index))
        timer.start()
        self._last_toggle = (index, timer)

    def _on_double_clicked(self, index: QModelIndex) -> None:
        group = self.findings_model().group_at(index)
        if group is not None:
            self.activated_group.emit(group.key)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            group = self.findings_model().group_at(self.currentIndex())
            if group is not None:
                self.activated_group.emit(group.key)
                event.accept()
                return
        super().keyPressEvent(event)


class FindingsList(QWidget):
    """Search, the painted list, and the selected finding in full.

    ``activated(key)`` when a file is opened (double click, Return, or the
    Open button under the detail); ``fix_requested(key, field)`` from the
    detail's Fix button.
    """

    activated = pyqtSignal(object)
    fix_requested = pyqtSignal(object, str)

    #: At or under this many findings the files open unfolded.
    UNFOLDED = 400

    def __init__(self, placeholder: str = "Find a file, a rule or a word",
                 parent=None) -> None:
        super().__init__(parent)
        from ..viz import fonts
        from .search_field import SearchField, find_shortcut

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(fonts.px(6))

        bar = QHBoxLayout()
        bar.setSpacing(fonts.px(8))
        self._search = SearchField(placeholder)
        bar.addWidget(self._search, 1)
        self._count = QLabel("")
        self._count.setObjectName("dlg-hint")
        bar.addWidget(self._count)
        v.addLayout(bar)
        find_shortcut(self, self._search)

        self.view = FindingsView(self)
        self.view.activated_group.connect(self.activated)
        v.addWidget(self.view, 1)

        self._detail = QWidget(self)
        self._detail.setObjectName("findings-detail")
        self._detail_layout = QVBoxLayout(self._detail)
        self._detail_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_layout.setSpacing(fonts.px(4))
        self._detail_hint = QLabel("Select a finding to read it in full. Double-click a "
                                   "file, or press Return, to open it.")
        self._detail_hint.setObjectName("dlg-hint")
        self._detail_hint.setWordWrap(True)
        self._detail_layout.addWidget(self._detail_hint)
        self._message: Optional[QWidget] = None
        v.addWidget(self._detail)

        # A search over 287,000 findings is a few tens of milliseconds;
        # waiting for the typing to pause keeps it from running per letter.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._apply_search)
        self._search.textChanged.connect(lambda _t: self._debounce.start())

        self.view.selectionModel().currentChanged.connect(self._on_current)

    # -- content --------------------------------------------------------

    def set_groups(self, groups: list[Group]) -> None:
        self.view.findings_model().set_groups(groups)
        self._after_reset()

    def _apply_search(self) -> None:
        self.view.findings_model().set_query(self._search.text())
        self._after_reset()

    def _after_reset(self) -> None:
        model = self.view.findings_model()
        groups, findings = model.counts()
        total_groups, total = model.total()
        if findings <= self.UNFOLDED:
            self.view.expandAll()
        word = "finding" if total == 1 else "findings"
        self._count.setText(f"{findings:,} {word}" if findings == total
                            else f"{findings:,} of {total:,} {word}")
        self._show(None, None)

    def search_text(self) -> str:
        return self._search.text()

    def set_search(self, text: str) -> None:
        """Search now, without waiting for typing to pause."""
        self._search.setText(text)
        self._debounce.stop()
        self._apply_search()

    # -- the selected finding ---------------------------------------------

    def _on_current(self, current: QModelIndex, _previous: QModelIndex) -> None:
        model = self.view.findings_model()
        self._show(model.group_at(current), model.finding_at(current))

    def _show(self, group: Optional[Group], finding: Optional[Finding]) -> None:
        from .val_message import ValMessage

        if self._message is not None:
            self._message.setParent(None)
            self._message.deleteLater()
            self._message = None
        if finding is None:
            self._detail_hint.setVisible(True)
            return
        self._detail_hint.setVisible(False)
        message = ValMessage(finding.severity, finding.rule, finding.message,
                             finding.fix_label, finding.field, finding.schema_rule,
                             parent=self._detail)
        message.fix_requested.connect(
            lambda field, key=group.key: self.fix_requested.emit(key, field))
        self._detail_layout.addWidget(message)
        self._message = message

    def current_finding(self) -> Optional[Finding]:
        return self.view.findings_model().finding_at(self.view.currentIndex())


__all__ = ["FINDING_ROLE", "Finding", "FindingsList", "FindingsModel", "FindingsView",
           "GROUP_ROLE", "Group", "KIND_ROLE", "plain"]

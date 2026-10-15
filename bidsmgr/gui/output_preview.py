"""Filter panel > BIDS preview: the dataset the conversion will write.

Every file the conversion and the metadata step will produce, as the
converters name them (:mod:`bidsmgr.converter.output_plan`): the data, its
sidecar, ``.bval``/``.bvec``, MNE-BIDS' channel and event tables, an eye
tracker's files per eye, each session's ``*_scans.tsv``, the dataset files,
a ``sourcedata/`` copy when the setting asks for one. A file written only
sometimes is shown dim and italic, with when. Below, the file picked: what it
will roughly contain, each value coloured by where it comes from (read from
the header during the scan, written by BIDS Manager from the settings, read
by the converter at conversion, or a TODO placeholder).

The plan and a file's preview are computed on a ``QThread`` (``JobRunner``)
from a snapshot of the table, only while this view is shown; the tree is
filled a folder at a time as it is opened, so a study of thousands of
recordings costs what is on screen.
"""

from __future__ import annotations

import html
import json
from typing import Callable, Optional

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QLabel,
    QStackedWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..converter.output_plan import UNKNOWN, OutputPlan, PlannedFile, approximate, plan_output
from . import icons
from .dialog_chrome import WrapLabel
from .theme_manager import CUR, scaled_px
from .widgets.flow_layout import FlowBar
from .widgets.primitives import ElidedLabel, qcolor
from .widgets.splitter import Splitter
from .widgets.tree_click import toggle_on_click

_PATH_ROLE = Qt.ItemDataRole.UserRole + 30
_PLACEHOLDER_ROLE = Qt.ItemDataRole.UserRole + 31
_SETTLE_MS = 150
#: What each source is called in the legend, and its palette token.
_LEGEND = (
    ("scan", "read by the scan", "text"),
    ("settings", "written by BIDS Manager", "accent"),
    ("stated", "your answers", "purple"),
    ("converter", f"{UNKNOWN} read at conversion", "dim"),
    ("todo", "TODO placeholder", "warning"),
)
_SOURCE_TOKEN = {key: token for key, _t, token in _LEGEND}


def _colour(source: str, pal: dict) -> str:
    """A source's colour in this theme.

    The accent is the user's to choose and can be a purple, which would make
    "written by BIDS Manager" and "your answers" one colour; then the answers
    take the teal.
    """
    token = _SOURCE_TOKEN.get(source, "text")
    if source == "stated":
        accent, purple = qcolor(pal["accent"]), qcolor(pal["purple"])
        gap = abs(accent.hslHue() - purple.hslHue())
        if accent.hslSaturation() > 40 and min(gap, 360 - gap) < 45:
            token = "teal"
    return pal[token]


class _LegendDot(QWidget):
    """A dot in a source's colour and its name."""

    def __init__(self, source: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self._source, self._text = source, text

    def _font(self) -> QFont:
        from .viz import fonts

        return fonts.font(10)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        m = QFontMetrics(self._font())
        return QSize(scaled_px(10) + m.horizontalAdvance(self._text) + 2, m.height() + 2)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        pal = CUR()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        d = scaled_px(6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(_colour(self._source, pal)))
        p.drawEllipse(0, (self.height() - d) // 2, d, d)
        p.setFont(self._font())
        p.setPen(qcolor(pal["dim"]))
        p.drawText(scaled_px(10), 0, self.width(), self.height(),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self._text)
        p.end()


class OutputPreview(QWidget):
    """The predicted BIDS tree and the predicted content of a file."""

    #: An inventory row whose file was picked (the table follows).
    row_chosen = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("output-preview")
        self._model = None
        self._context: Optional[Callable[[], tuple]] = None
        self._plan: Optional[OutputPlan] = None
        self._snapshot = None
        self._dirty = True
        self._jobs = None
        self._ticket = 0
        self._file_ticket = 0
        self._folders: dict[int, dict] = {}
        self._revealing = False

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._start)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack)
        self._empty = QLabel("Scan first: the preview shows what the conversion will write.", self)
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._stack.addWidget(self._empty)

        split = Splitter(Qt.Orientation.Vertical, self)
        split.setChildrenCollapsible(False)
        self.tree = QTreeWidget(split)
        self.tree.setObjectName("plan-tree")
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(14)
        self.tree.itemExpanded.connect(self._fill)
        self.tree.currentItemChanged.connect(self._on_current)
        # One click on a folder row opens or closes it, as in every other tree.
        toggle_on_click(self.tree)
        split.addWidget(self.tree)

        box = QWidget(split)
        box.setObjectName("plan-file")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(4)
        self.file_name = ElidedLabel("", box)
        self.file_name.setObjectName("plan-file-name")
        bl.addWidget(self.file_name)
        self.file_origin = QLabel("", box)
        self.file_origin.setObjectName("dlg-hint")
        bl.addWidget(self.file_origin)
        self.content = QTextBrowser(box)
        self.content.setObjectName("plan-preview")
        self.content.setOpenLinks(False)
        self.content.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        bl.addWidget(self.content, 1)
        self.file_note = WrapLabel("", box)
        self.file_note.setObjectName("dlg-hint")
        self.file_note.setTextFormat(Qt.TextFormat.PlainText)
        bl.addWidget(self.file_note)
        legend = FlowBar(box, h_spacing=10, v_spacing=2)
        for key, text, _token in _LEGEND:
            legend.addWidget(_LegendDot(key, text, legend))
        bl.addWidget(legend)
        split.addWidget(box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self._split = split
        self._stack.addWidget(split)
        self._show_file(None)

    # -- binding ------------------------------------------------------------------

    def bind_model(self, model) -> None:
        if model is self._model:
            return
        if self._model is not None:
            for signal in (self._model.dataChanged, self._model.modelReset,
                           self._model.recordingSpecChanged):
                try:
                    signal.disconnect(self.refresh)
                except (TypeError, RuntimeError):
                    pass
        self._model = model
        self._plan = None
        if model is not None:
            model.dataChanged.connect(self.refresh)
            model.modelReset.connect(self.refresh)
            # A per-file answer or a template changes what a file holds and
            # touches no cell of the table.
            model.recordingSpecChanged.connect(self.refresh)
        self.refresh()

    def set_context(self, provider: Callable[[], tuple]) -> None:
        """``provider()`` returns ``(PlanOptions, features, spec, first
        files)``: the settings, the scan's series features, the recording
        metadata, and each series' first DICOM by uid."""
        self._context = provider
        self.refresh()

    def refresh(self, *_args) -> None:
        """The table or the settings changed: plan again when shown."""
        self._dirty = True
        if self.isVisible():
            self._settle.start()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        if self._dirty:
            self._settle.start(0)

    def plan(self) -> Optional[OutputPlan]:
        return self._plan

    def is_planning(self) -> bool:
        return self._jobs is not None and self._jobs.running("plan")

    # -- planning -----------------------------------------------------------------

    def _runner(self):
        if self._jobs is None:
            from .viz.bridge import JobRunner

            self._jobs = JobRunner(self)
            self._jobs.done.connect(self._on_done)
            self._jobs.failed.connect(self._on_failed)
        return self._jobs

    def _start(self) -> None:
        model = self._model
        self._dirty = False
        if model is None or model.rowCount() == 0:
            self._plan = None
            self._empty.setText("Scan first: the preview shows what the conversion will write.")
            self._stack.setCurrentIndex(0)
            return
        options, features, spec, first_files = (self._context() if self._context
                                                else (None, None, None, None))
        from ..converter.output_plan import PlanOptions

        options = options or PlanOptions()
        snapshot = model.dataframe().copy()
        included = [model._read_include(r) for r in range(model.rowCount())]
        self._snapshot = (snapshot, included, options, features, spec)
        self._ticket = self._ticket % 1_000_000 + 1
        self._runner().start("plan", self._ticket, plan_output, snapshot, included, options,
                             features, spec, first_files)

    def _on_done(self, tag: str, ticket: int, result) -> None:
        if tag == "plan":
            if ticket != self._ticket:
                return
            self._plan = result
            self._build_tree()
        elif tag == "file" and ticket == self._file_ticket:
            planned, preview = result
            self._render(planned, preview)

    def _on_failed(self, tag: str, _ticket: int, message: str) -> None:
        if tag == "plan":
            self._empty.setText(f"The preview could not be made: {message}")
            self._stack.setCurrentIndex(0)
        else:
            self.content.setPlainText(f"Could not predict this file: {message}")

    # -- the tree -----------------------------------------------------------------

    def _build_tree(self) -> None:
        plan = self._plan
        current = self._current_path()
        self.tree.clear()
        self._folders = {}
        if plan is None or not plan.files:
            self._empty.setText("Nothing is set to convert.")
            self._stack.setCurrentIndex(0)
            return
        root: dict = {"dirs": {}, "files": []}
        for f in plan.files:
            parts = f.path.rstrip("/").split("/")
            node = root
            for part in parts[:-1]:
                node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})
            if f.path.endswith("/"):
                node["dirs"].setdefault(parts[-1], {"dirs": {}, "files": [], "planned": f})
            else:
                node["files"].append(f)
        self._add_children(None, root, "")
        self._stack.setCurrentIndex(1)
        first_subject = next((self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
                              if self.tree.topLevelItem(i).text(0).startswith("sub-")), None)
        if first_subject is not None:
            first_subject.setExpanded(True)
        if current:
            self.select_path(current)

    def _add_children(self, parent: Optional[QTreeWidgetItem], node: dict, prefix: str) -> None:
        pal = CUR()
        dirs = sorted(node["dirs"].items(), key=lambda kv: (kv[0] == "sourcedata", kv[0].lower()))
        top_level = parent is None
        # Alphabetical as a file browser sorts: case does not come first.
        files = sorted(node["files"], key=lambda f: f.path.lower())
        # Dataset files first at the top, then the subjects; inside a folder,
        # folders first.
        order = ([("file", f) for f in files] + [("dir", d) for d in dirs]) if top_level \
            else ([("dir", d) for d in dirs] + [("file", f) for f in files])
        for kind, entry in order:
            if kind == "dir":
                name, child = entry
                item = QTreeWidgetItem([name])
                item.setIcon(0, icons.icon_for_path(name, is_dir=True))
                item.setForeground(0, QBrush(qcolor(pal[icons.color_token_for(name, is_dir=True)])))
                path = f"{prefix}{name}/"
                item.setData(0, _PATH_ROLE, path)
                planned = child.get("planned")
                if planned is not None:
                    item.setToolTip(0, f"{path}\nWritten by {planned.origin}.")
                if child["dirs"] or child["files"]:
                    placeholder = QTreeWidgetItem([""])
                    placeholder.setData(0, _PLACEHOLDER_ROLE, True)
                    item.addChild(placeholder)
                    self._folders[id(item)] = (child, path)
            else:
                f: PlannedFile = entry
                name = f.path.rsplit("/", 1)[-1]
                item = QTreeWidgetItem([name])
                item.setIcon(0, icons.icon_for_path(name))
                item.setData(0, _PATH_ROLE, f.path)
                tip = f"{f.path}\nWritten by {f.origin}."
                if f.warning or f.renamed:
                    # Written, but not a name BIDS allows, or not the name in
                    # the table: say so where it is.
                    item.setForeground(0, QBrush(qcolor(pal["warning"])))
                    item.setIcon(0, icons.icon("warn"))
                    tip += (f"\nNot valid BIDS: {f.warning}" if f.warning
                            else f"\n{f.renamed}")
                elif f.conditional:
                    font = item.font(0)
                    font.setItalic(True)
                    item.setFont(0, font)
                    item.setForeground(0, QBrush(qcolor(pal["dim"])))
                    tip += f"\nOnly {f.conditional}."
                else:
                    item.setForeground(0, QBrush(qcolor(pal[icons.color_token_for(name)])))
                item.setToolTip(0, tip)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)

    def _fill(self, item: QTreeWidgetItem) -> None:
        """Draw a folder's entries the first time it is opened."""
        slot = self._folders.pop(id(item), None)
        if slot is None:
            return
        node, path = slot
        for i in reversed(range(item.childCount())):
            if item.child(i).data(0, _PLACEHOLDER_ROLE):
                item.takeChild(i)
        self._add_children(item, node, path)
        if item.childCount() == 1 and item.child(0).childCount():
            item.child(0).setExpanded(True)

    def _current_path(self) -> str:
        item = self.tree.currentItem()
        return str(item.data(0, _PATH_ROLE) or "") if item is not None else ""

    def select_path(self, path: str) -> bool:
        """Open the folders down to ``path`` and make it current."""
        parts = path.rstrip("/").split("/")
        items = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        item = None
        for depth, part in enumerate(parts):
            item = next((x for x in items if x.text(0) == part), None)
            if item is None:
                return False
            if depth < len(parts) - 1:
                item.setExpanded(True)
                items = [item.child(i) for i in range(item.childCount())]
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        return True

    def reveal_row(self, row: Optional[int]) -> None:
        """Show the data file of inventory ``row`` (the table's selection),
        unless a file of that row is already the one shown."""
        if row is None or self._plan is None:
            return
        current = self._plan.find(self._current_path())
        if current is not None and row in current.rows:
            return
        files = [f for f in self._plan.of_row(row) if f.role == "data" and not f.conditional]
        if files:
            self._revealing = True
            try:
                self.select_path(files[0].path)
            finally:
                self._revealing = False

    # -- one file -------------------------------------------------------------------

    def _on_current(self, current: Optional[QTreeWidgetItem], _previous) -> None:
        path = str(current.data(0, _PATH_ROLE) or "") if current is not None else ""
        planned = self._plan.find(path) if (self._plan is not None and path) else None
        if planned is None:
            self._show_file(None, folder=path)
            return
        if planned.rows and not self._revealing:
            self.row_chosen.emit(int(planned.rows[0]))
        snapshot, included, options, features, spec = self._snapshot
        self._file_ticket = self._file_ticket % 1_000_000 + 1
        self.file_name.setText(path.rsplit("/", 1)[-1])
        self._runner().start("file", self._file_ticket, _preview_job, snapshot, planned, options,
                             features, spec, included)

    def _show_file(self, planned: Optional[PlannedFile], folder: str = "") -> None:
        self.file_name.setText(folder.rstrip("/").rsplit("/", 1)[-1] if folder else
                               "Pick a file to see what it will contain")
        self.file_origin.setText("")
        self.content.setHtml("")
        self.file_note.setText("")
        self.file_note.setVisible(False)

    def _render(self, planned: PlannedFile, preview) -> None:
        pal = CUR()
        self.file_name.setText(planned.path.rsplit("/", 1)[-1])
        origin = f"Written by {planned.origin}"
        if planned.conditional:
            origin += f", only {planned.conditional}"
        self.file_origin.setText(origin + ". A prediction.")
        self.content.setHtml(_html(preview, pal))
        note = " ".join(x for x in (f"Not valid BIDS: {planned.warning}" if planned.warning else "",
                                    planned.renamed, preview.note) if x)
        self.file_note.setText(note)
        self.file_note.setVisible(bool(note))

    def repaint_for_palette(self, _pal: dict) -> None:
        """Rebuild what holds palette colours: the tree's entries and the
        coloured preview."""
        if self._plan is not None:
            self._build_tree()
        item = self.tree.currentItem()
        if item is not None:
            self._on_current(item, None)


def _preview_job(df, planned, options, features, spec, included):
    return planned, approximate(df, planned, options, features, spec, included)


def _span(text: str, colour: str, *, bold: bool = False) -> str:
    weight = "font-weight:600;" if bold else ""
    return f'<span style="color:{colour};{weight}">{html.escape(text)}</span>'


def _html(preview, pal: dict) -> str:
    """The predicted file as it would read, coloured by source."""
    lines: list[str] = []
    if preview.kind == "json":
        lines.append(_span("{", pal["dim"]))
        n = len(preview.fields)
        for k, (key, value, source) in enumerate(preview.fields):
            colour = _colour(source, pal)
            text = json.dumps(value, ensure_ascii=False)
            comma = "," if k < n - 1 else ""
            lines.append("&nbsp;&nbsp;" + _span(json.dumps(key), pal["text"]) + _span(": ", pal["dim"])
                         + _span(text, colour) + _span(comma, pal["dim"]))
        lines.append(_span("}", pal["dim"]))
    elif preview.kind == "tsv":
        widths = [len(h) for h in preview.header]
        for row in preview.rows:
            for c, cell in enumerate(row[:len(widths)]):
                widths[c] = max(widths[c], len(str(cell)))
        def fmt(cells, colour, bold=False):
            padded = "&nbsp;&nbsp;".join(
                _span(str(cell), colour if str(cell) != UNKNOWN else pal["dim"], bold=bold)
                + "&nbsp;" * (widths[c] - len(str(cell))) for c, cell in enumerate(cells))
            return padded
        if preview.header:
            lines.append(fmt(preview.header, pal["text"], bold=True))
        for row in preview.rows:
            lines.append(fmt(row, pal["text"]))
        if not preview.rows:
            lines.append(_span(UNKNOWN, pal["dim"]))
    elif preview.kind == "text":
        lines = [html.escape(line) or "&nbsp;" for line in preview.text.splitlines()]
    elif preview.kind == "folder":
        lines = [_span("A folder of the source files, as they are.", pal["dim"])]
    else:
        lines = [_span("Binary data: nothing to read here.", pal["dim"])]
    body = "<br>".join(lines)
    return f'<div style="white-space:pre; color:{pal["text"]};">{body}</div>'


__all__ = ["OutputPreview"]

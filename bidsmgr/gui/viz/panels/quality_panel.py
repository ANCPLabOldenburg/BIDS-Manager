"""The quality panel: the check of an anatomical or diffusion image
(``bidsmgr.qc``), docked in the viewer like the time course.

Beside the views by default (a report is a tall list), below them on
request, maximised, or in a window of its own: the same corner the time
course has. For a diffusion series the per-volume QC plots stay under the
time course, on the volume axis they share, and this panel holds the rest.

In the order a reviewer needs it:

1. **On the image**: one checkbox per mask or map the check made (the brain,
   CSF, grey matter and white matter each, the air it measured, the
   artefacts in it, the bias field...), each with its colour, Show the
   noise, and for a diffusion series the QC plots. A checkbox shows and
   hides ONE layer; it never adds a second.
2. **Findings**, each with its evidence a click away (Show, Go there).
3. **Measures**, grouped (each group folds away), with where each sits
   among the dataset's checked images of the same kind. What a measure
   means is on hover, and in full behind its info icon
   (``explain_popup``, from ``bidsmgr.qc.explain``), never in a box the
   reader has to scroll to.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QEvent, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QBoxLayout, QCheckBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ....qc.types import QCResult
from ...widgets.flow_layout import FlowBar
from .. import fonts
from ..bridge import ThemeHub, connect_while_alive
from .panel_header import PanelHeader, corner_button

#: Report sections, in order, as the reader meets them.
GROUPS = (
    ("gradients", "Gradient table"),
    ("motion", "Motion"),
    ("noise", "Noise and contrast"),
    ("artefacts", "Artefacts"),
    ("tissues", "Tissues"),
    ("diffusion", "Diffusion"),
    ("coverage", "Coverage and smoothness"),
    ("header", "Header"),
)
_LEVEL_CHIP = {"error": ("Error", "err"), "warning": ("Warning", "warn"),
               "info": ("Note", "accent"), "ok": ("OK", "success")}
#: Beyond this robust z within the dataset, a value is flagged.
FLAG_Z = 3.0
#: The colours tissue classes are read in: CSF blue, grey matter green,
#: white matter yellow.
TISSUE_COLOURS = {1: (59, 130, 246), 2: (34, 197, 94), 3: (234, 179, 8)}


def label_colours(key: str, qmap) -> dict:
    """``{label value: (r, g, b)}`` for a labelled map."""
    if key == "tissues":
        return dict(TISSUE_COLOURS)
    from ....viz.data.labels import generated_table

    return dict(generated_table(sorted(qmap.labels)).colors)


def split_key(key: str) -> tuple[str, Optional[int]]:
    """``"tissues:2"`` -> ``("tissues", 2)``: one class of a labelled map;
    ``"brain"`` -> ``("brain", None)``."""
    base, _sep, value = key.partition(":")
    return base, (int(value) if value else None)


def evidence_items(result: QCResult) -> list[tuple[str, str, str, object]]:
    """What can go on the image, in order: ``(key, title, help, colour)``
    with ``colour`` an (r, g, b) or a colour map name. A labelled map is one
    item PER CLASS (CSF, grey matter, white matter each its own layer)."""
    out = []
    for key, qmap in result.maps.items():
        if qmap.kind == "labels":
            colours = label_colours(key, qmap)
            for value, name in sorted(qmap.labels.items()):
                out.append((f"{key}:{int(value)}", str(name), qmap.help,
                            tuple(colours.get(int(value), (128, 128, 128)))[:3]))
        else:
            out.append((key, qmap.title, qmap.help,
                        qmap.colour if qmap.kind == "field" else _mask_colour(qmap.colour)))
    return out


def _mask_colour(name: str) -> tuple:
    """A mask's colour: its colour map at the top, where a mask's 1 lands."""
    from ....viz.compute import colormaps

    return tuple(int(c) for c in colormaps.lut(name or "red")[255][:3])


def _swatch(colour, size: QSize, ratio: float) -> QIcon:
    """A legend mark: a DOT of a solid colour (round, so it never reads as
    a second checkbox beside the real one), or a colour map's strip with
    round ends."""
    w, h = max(1, round(size.width() * ratio)), max(1, round(size.height() * ratio))
    if isinstance(colour, str):
        from ....viz.compute import colormaps

        strip = np.ascontiguousarray(np.repeat(colormaps.swatch(colour, w), h, axis=0))
        fill = QPixmap.fromImage(QImage(strip.data, w, h, w * 4,
                                        QImage.Format.Format_RGBA8888).copy())
    else:
        fill = QPixmap(w, h)
        fill.fill(QColor(*colour))
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    if isinstance(colour, str):
        path.addRoundedRect(QRectF(0, 0, w, h), h / 2.0, h / 2.0)
    else:
        path.addEllipse(QRectF(0, 0, w, h))
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, fill)
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return QIcon(pix)


#: Columns of the measures table.
COL_NAME, COL_VALUE, COL_DATASET, COL_INFO = range(4)


#: Wider than this many average characters (below the views, maximised, its
#: own window), the measures go in a column of their own beside the rest.
TWO_COLUMNS_CHARS = 110


def _item(text: str, *, tip: str = "", data=None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        item.setToolTip(tip)
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    return item


def dataset_text(z: Optional[float], better: str) -> tuple[str, str]:
    """``(text, level)`` for a measure's place in the dataset: "typical",
    or how far off and which way, "warning" when that way is worse."""
    if z is None:
        return "", ""
    if abs(z) < 2.0:
        return "typical", ""
    side = "high" if z > 0 else "low"
    worse = (better == "higher" and z < 0) or (better == "lower" and z > 0)
    level = "warning" if worse and abs(z) >= FLAG_Z else ""
    return f"{side} (z {z:+.1f})", level


def summary_text(result: QCResult, compared: bool) -> str:
    """One line: what was checked, how, and whether the air was measured."""
    facts = result.facts
    parts = ["Diffusion series" if result.kind == "dwi" else (result.suffix or result.kind)]
    if facts.get("seconds") is not None:
        parts.append(f"checked in {facts['seconds']:.0f} s")
    if result.kind == "anat":
        if facts.get("air_measured"):
            parts.append("not defaced: the air was measured")
        elif facts.get("air_reason"):
            parts.append(facts["air_reason"].rstrip("."))
    if result.kind == "dwi" and facts.get("shells"):
        parts.append("volumes per shell " + ", ".join(
            f"b={k}: {v}" for k, v in facts["shells"].items()))
    parts.append("compared with the dataset's other checked images" if compared
                 else "no other checked image of its kind to compare with")
    return ". ".join(p[:1].upper() + p[1:] for p in parts if p) + "."


def engine_text(result: QCResult) -> str:
    """What made the masks: the tools, or the approximate fallback."""
    eng = result.facts.get("engine") or {}
    bits = []
    if eng.get("brain"):
        bits.append(f"brain: {eng['brain']}")
    if eng.get("tissues"):
        bits.append(f"tissues: {eng['tissues']}")
    if eng.get("registration"):
        bits.append(f"registration: {eng['registration']}")
    text = "Made with " + "; ".join(bits) + "." if bits else ""
    note = result.facts.get("tool_note")
    if note:
        text += f" The tools could not run ({note}): the masks are approximate."
    return text


class QualityPanel(QWidget):
    """The quality check of the image on screen, docked in the viewer."""

    #: Check the image on screen (from the empty state, or More > Check again).
    check_requested = pyqtSignal()
    #: Show (True) or hide (False) one of the check's maps on the image.
    show_map = pyqtSignal(str, bool)
    #: Go to a finding's evidence: {"volume": int, "slice": int | None, "axis"}.
    go_to = pyqtSignal(object)
    #: Window the image to its air (True) or back (False).
    noise = pyqtSignal(bool)
    #: The diffusion QC plots under the time course.
    plots = pyqtSignal(bool)
    #: Save the result into the dataset's QC derivative.
    save = pyqtSignal()
    #: The panel's close button.
    close_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viz-quality")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.result: Optional[QCResult] = None
        self.context: dict = {}
        self._rows: list = []
        self.map_boxes: dict[str, QCheckBox] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = PanelHeader()
        title = QLabel("Quality")
        title.setObjectName("viz-track-title")
        header.controls.addWidget(title)
        # The image it describes: always the one on screen, named by the
        # status bar (and by the window's title when it has its own).
        self.name = ""
        self.more_button, more = header.more_menu("More: save the result, check again")
        self.save_action = more.addAction("Save to dataset")
        self.save_action.setToolTip("Write the result into derivatives/bidsmgr-qc/, where the "
                                    "Editor's Quality check and the next look at this image "
                                    "find it")
        self.save_action.triggered.connect(self.save.emit)
        self.again_action = more.addAction("Check again")
        self.again_action.setToolTip("Compute the check again (after the image changed)")
        self.again_action.triggered.connect(self.check_requested.emit)
        self.close_button = header.add_corner(corner_button("close", "Close the quality panel"))
        self.close_button.clicked.connect(self.close_requested.emit)
        self.header = header
        lay.addWidget(header)

        scroll = QScrollArea()
        scroll.setObjectName("viz-tracks-scroll")
        scroll.viewport().setObjectName("viz-tracks-viewport")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("viz-tracks-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Two columns, stacked when narrow: what is on the image and the
        # findings, then the measures.
        self.columns = QBoxLayout(QBoxLayout.Direction.TopToBottom, body)
        self.columns.setContentsMargins(10, 8, 10, 10)
        self.columns.setSpacing(12)
        layouts = []
        for _ in range(2):
            col = QWidget()
            col.setObjectName("viz-tracks-body")
            v = QVBoxLayout(col)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(8)
            self.columns.addWidget(col)
            layouts.append(v)
        self.body_layout, self.measures_layout = layouts
        self.columns.setStretch(1, 1)
        scroll.setWidget(body)
        self.scroll = scroll
        lay.addWidget(scroll, 1)
        connect_while_alive(ThemeHub.instance().changed, self, QualityPanel._recolour)
        self.set_empty("")

    # -- states -----------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        super().resizeEvent(event)
        wide = self.width() >= self.fontMetrics().averageCharWidth() * TWO_COLUMNS_CHARS
        direction = (QBoxLayout.Direction.LeftToRight if wide
                     else QBoxLayout.Direction.TopToBottom)
        if self.columns.direction() != direction:
            self.columns.setDirection(direction)
            self.columns.setStretch(0, 2 if wide else 0)
            self.columns.setStretch(1, 3 if wide else 1)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt signature
        # The VIEWPORT's resize: the table's own arrives before Qt resizes
        # the viewport, so a fit there used the width before.
        if (event.type() == QEvent.Type.Resize and self.table is not None
                and obj is self.table.viewport()):
            self._fit_columns()
        return super().eventFilter(obj, event)

    def _clear(self) -> None:
        for layout in (self.body_layout, self.measures_layout):
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
        self.map_boxes = {}
        self.noise_box = None
        self.plots_box = None
        self.table = None
        self._rows = []

    def _note(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("viz-track-readout")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        self.body_layout.addWidget(label)
        return label

    def set_empty(self, name: str, *, reason: str = "") -> None:
        """No result for the image on screen: what the check is, and a
        button to run it (none when it cannot run on this image)."""
        self.result = None
        self._clear()
        self.name = name
        self._set_more(False)
        self._note(reason or "The fast quality check of an anatomical or diffusion image: "
                   "noise, contrast, artefacts, tissues, coverage, and for diffusion the "
                   "gradient table, motion and slice dropout, each finding with its "
                   "evidence on the image. The air is measured only in images that are "
                   "not defaced.")
        if not reason:
            btn = QPushButton("Check quality")
            btn.setObjectName("tb-btn-primary")
            btn.clicked.connect(self.check_requested.emit)
            row = QHBoxLayout()
            row.addWidget(btn)
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            self.body_layout.addWidget(holder)
        self.body_layout.addStretch(1)

    def set_busy(self, message: str) -> None:
        """The check is running."""
        from ...widgets.spinner import BusySpinner

        self.result = None
        self._clear()
        self._set_more(False)
        # Placed, then shown: a spinner shown before it had a parent was a
        # window for a moment and took the focus from the preview on macOS.
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        spinner = BusySpinner(holder)
        row.addWidget(spinner)
        self.busy_label = QLabel(message, holder)
        self.busy_label.setObjectName("viz-track-readout")
        self.busy_label.setWordWrap(True)
        row.addWidget(self.busy_label, 1)
        self.body_layout.addWidget(holder)
        self.body_layout.addStretch(1)
        spinner.set_busy(True, message="")

    def set_progress(self, message: str) -> None:
        label = getattr(self, "busy_label", None)
        if label is not None:
            try:
                label.setText(message)
            except RuntimeError:        # replaced since
                pass

    def _set_more(self, have: bool) -> None:
        self.save_action.setEnabled(have and self._can_save)
        self.again_action.setEnabled(have)

    _can_save = True

    def set_result(self, result: QCResult, name: str, *, context: Optional[dict] = None,
                   shown: Optional[set] = None, noise_on: bool = False, plots_on: bool = False,
                   can_save: bool = True) -> None:
        """Show ``result``: what is on the image, the findings, the measures."""
        self.result = result
        self.context = dict(context or {})
        self._can_save = can_save
        self._clear()
        self._set_more(True)
        self.name = name
        self.summary = self._note(summary_text(result, bool(self.context)))
        engine = engine_text(result)
        if engine:
            self._note(engine)
        self._build_on_image(result, shown or set(), noise_on, plots_on)
        self._build_findings(result)
        self.body_layout.addStretch(1)
        heading = QLabel("Measures")
        heading.setObjectName("viz-track-title")
        self.measures_layout.addWidget(heading)
        hint = QLabel("Hover a measure for what it is; its info icon explains it and how "
                      "to read it. Click a group's name to fold it.")
        hint.setObjectName("viz-track-readout")
        hint.setWordWrap(True)
        self.measures_layout.addWidget(hint)
        self.table = self._metric_table(result)
        self.measures_layout.addWidget(self.table)
        self.measures_layout.addStretch(1)

    # -- on the image -------------------------------------------------------------

    def _build_on_image(self, result: QCResult, shown: set, noise_on: bool,
                        plots_on: bool) -> None:
        card = QFrame()
        card.setObjectName("viz-track")
        col = QVBoxLayout(card)
        col.setContentsMargins(10, 6, 10, 8)
        col.setSpacing(4)
        title = QLabel("On the image")
        title.setObjectName("viz-track-title")
        col.addWidget(title)
        bar = FlowBar(h_spacing=12, v_spacing=4)
        side = max(8, round(self.fontMetrics().height() * 0.6))
        ratio = self.devicePixelRatioF()
        for key, title, tip, colour in evidence_items(result):
            box = QCheckBox(title)
            box.setToolTip(tip)
            # Its colour on the image, beside its name: a solid swatch for a
            # mask or a class, a strip for a field.
            size = QSize(side * 2 if isinstance(colour, str) else side, side)
            box.setIcon(_swatch(colour, size, ratio))
            box.setIconSize(size)
            box.setChecked(key in shown)
            box.toggled.connect(lambda on, k=key: self.show_map.emit(k, bool(on)))
            bar.addWidget(box)
            self.map_boxes[key] = box
        self.noise_box = QCheckBox("Show the noise")
        self.noise_box.setToolTip("The image windowed to its air, in colour: ghosts, ringing, "
                                  "motion and wrap-around show as structure where there "
                                  "should be noise")
        self.noise_box.setChecked(noise_on)
        self.noise_box.toggled.connect(lambda on: self.noise.emit(bool(on)))
        bar.addWidget(self.noise_box)
        if result.kind == "dwi":
            self.plots_box = QCheckBox("QC plots under the time course")
            self.plots_box.setToolTip("Displacement, the slice signal against the tensor, the "
                                      "b=0 signal and spikes, per volume; click a slice to go "
                                      "there")
            self.plots_box.setChecked(plots_on)
            self.plots_box.toggled.connect(lambda on: self.plots.emit(bool(on)))
            bar.addWidget(self.plots_box)
        col.addWidget(bar)
        self.body_layout.addWidget(card)

    def _set_box(self, box: Optional[QCheckBox], on: bool) -> None:
        if box is not None and box.isChecked() != bool(on):
            box.blockSignals(True)
            box.setChecked(bool(on))
            box.blockSignals(False)

    def set_map_shown(self, key: str, on: bool) -> None:
        self._set_box(self.map_boxes.get(key), on)

    def set_noise(self, on: bool) -> None:
        self._set_box(self.noise_box, on)

    def set_plots(self, on: bool) -> None:
        self._set_box(self.plots_box, on)

    # -- findings ---------------------------------------------------------------

    def _build_findings(self, result: QCResult) -> None:
        from ...widgets.primitives import Chip

        order = {"error": 0, "warning": 1, "info": 2, "ok": 3}
        findings = sorted(result.findings, key=lambda f: order.get(f.level, 9))
        if not any(f.level in ("warning", "error") for f in findings):
            findings = [type("F", (), {"key": "none", "title": "Nothing to flag",
                                       "level": "ok", "evidence": None,
                                       "message": "No measure or check calls for a look."})()
                        ] + findings
        self.finding_cards = []
        for f in findings:
            card = QFrame()
            card.setObjectName("viz-track")
            row = QHBoxLayout(card)
            row.setContentsMargins(10, 6, 8, 6)
            row.setSpacing(8)
            text, kind = _LEVEL_CHIP.get(f.level, ("Note", "default"))
            row.addWidget(Chip(text, kind), 0, Qt.AlignmentFlag.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(1)
            title = QLabel(f.title)
            title.setObjectName("viz-track-title")
            title.setWordWrap(True)
            col.addWidget(title)
            msg = QLabel(f.message)
            msg.setObjectName("viz-track-readout")
            msg.setWordWrap(True)
            msg.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            msg.setMinimumWidth(80)
            col.addWidget(msg)
            row.addLayout(col, 1)
            button = self._evidence_button(f.evidence)
            if button is not None:
                row.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
            self.body_layout.addWidget(card)
            self.finding_cards.append(card)

    def _evidence_button(self, evidence) -> Optional[QPushButton]:
        if isinstance(evidence, dict) and evidence.get("volume") is not None:
            btn = QPushButton("Go there")
            btn.setObjectName("tb-btn")
            where = f"volume {evidence['volume']}"
            if evidence.get("slice") is not None:
                where += f", slice {evidence['slice']}"
            btn.setToolTip(f"Show {where}")
            btn.clicked.connect(lambda _c=False, e=dict(evidence): self.go_to.emit(e))
            return btn
        if isinstance(evidence, str) and self.result is not None and evidence in self.result.maps:
            btn = QPushButton("Show")
            btn.setObjectName("tb-btn")
            btn.setToolTip(f"Show the {self.result.maps[evidence].title.lower()} on the image")
            btn.clicked.connect(lambda _c=False, k=evidence: self._show_evidence(k))
            return btn
        return None

    def _show_evidence(self, key: str) -> None:
        """Tick what a finding's evidence names: its box, or every class of
        a labelled map ("tissues" ticks CSF, grey and white matter)."""
        boxes = [b for k, b in self.map_boxes.items() if split_key(k)[0] == key]
        if not boxes:
            self.show_map.emit(key, True)
        for box in boxes:
            if not box.isChecked():
                box.setChecked(True)        # emits show_map

    # -- measures ---------------------------------------------------------------

    #: Groups the user folded, kept from one image to the next.
    _collapsed: set = set()

    def _metric_table(self, result: QCResult) -> QTableWidget:
        """One row per measure under its group's row: the name (hover: what
        it is), the value, where it sits in the dataset, and an info icon
        (the full explanation). A group's row folds the group away and has
        its own info icon (how its measures relate)."""
        from ... import icons
        from .explain_popup import hover_text

        self._collapsed = set(self._collapsed)
        self._kind = result.kind
        table = QTableWidget(0, 4, self)
        table.setObjectName("viz-quality-table")
        table.setHorizontalHeaderLabels(["Measure", "Value", "In this dataset", ""])
        table.horizontalHeaderItem(COL_DATASET).setToolTip(
            "Where the value sits among the dataset's checked images of the same kind and "
            "acquisition (robust z), and whether that is the worse side.")
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        hh = table.horizontalHeader()
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        by_group: dict[str, list] = {}
        for m in result.metrics:
            by_group.setdefault(m.group or "other", []).append(m)
        names = dict(GROUPS)
        ordered = [g for g, _t in GROUPS if g in by_group] + [
            g for g in by_group if g not in names]
        info_icon = icons.icon("info")
        for g in ordered:
            r = table.rowCount()
            table.insertRow(r)
            flagged = sum(1 for m in by_group[g]
                          if dataset_text(self.context.get(m.key), m.better)[1])
            text = names.get(g, g.capitalize())
            if flagged:
                text += f"  ({flagged} off in this dataset)"
            head = _item(text, tip="Click to fold or unfold this group.")
            font = head.font()
            font.setBold(True)
            head.setFont(font)
            head.setFlags(Qt.ItemFlag.ItemIsEnabled)
            table.setItem(r, COL_NAME, head)
            table.setSpan(r, COL_NAME, 1, 3)
            key = f"group.{result.kind}.{g}"
            info = _item("", tip=hover_text(key) or "What this group measures.")
            info.setIcon(info_icon)
            info.setFlags(Qt.ItemFlag.ItemIsEnabled)
            table.setItem(r, COL_INFO, info)
            self._rows.append(("group", g))
            for m in by_group[g]:
                r = table.rowCount()
                table.insertRow(r)
                key = f"{result.kind}.{m.key}"
                extra = []
                if m.value is None and m.why_missing:
                    extra.append(f"Not computed: {m.why_missing}")
                if m.mriqc:
                    extra.append(f"MRIQC's counterpart: {m.mriqc}.")
                tip = hover_text(key, better=m.better, extra=" ".join(extra)) or m.help
                name = _item(m.title, tip=tip, data=m.key)
                table.setItem(r, COL_NAME, name)
                value = _item(m.text(), tip=m.why_missing or tip)
                value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, COL_VALUE, value)
                text, _level = dataset_text(self.context.get(m.key), m.better)
                table.setItem(r, COL_DATASET, _item(
                    text, tip="Robust z among the dataset's checked images of the same kind "
                              "and acquisition" if text else ""))
                info = _item("", tip="What this measures and how to read it")
                info.setIcon(info_icon)
                table.setItem(r, COL_INFO, info)
                self._rows.append(m)
                table.setRowHidden(r, g in self._collapsed)
        table.cellClicked.connect(self._on_cell)
        table.cellDoubleClicked.connect(lambda r, _c: self._explain_row(r))
        # What each column needs; _fit_columns shares the room out.
        self._needs = []
        for c in range(4):
            table.resizeColumnToContents(c)
            self._needs.append(table.columnWidth(c) + 16)
        self._needs[COL_INFO] = fonts.px(30)
        # Nothing to compare with: no column of blanks.
        table.setColumnHidden(COL_DATASET, not self.context)
        table.viewport().installEventFilter(self)
        # The whole table, no scroll bar of its own: the panel scrolls once.
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.resizeRowsToContents()
        self._fit_height(table)
        # Coloured (bands, fold arrows, icons) once it is the panel's table.
        self.table = table
        self._recolour()
        return table

    def _fit_height(self, table: QTableWidget) -> None:
        table.setFixedHeight(table.horizontalHeader().sizeHint().height() + 4 + sum(
            table.rowHeight(r) for r in range(table.rowCount()) if not table.isRowHidden(r)))

    def _on_cell(self, row: int, col: int) -> None:
        entry = self._rows[row] if 0 <= row < len(self._rows) else None
        if isinstance(entry, tuple):
            if col == COL_INFO:
                self._explain_row(row)
            else:
                self.toggle_group(entry[1])
        elif entry is not None and col == COL_INFO:
            self._explain_row(row)

    def toggle_group(self, group: str) -> None:
        """Fold ``group`` away, or back."""
        table = self.table
        if table is None:
            return
        fold = group not in self._collapsed
        self._collapsed = (self._collapsed | {group}) if fold else (self._collapsed - {group})
        current = None
        for r, entry in enumerate(self._rows):
            if isinstance(entry, tuple):
                current = entry[1]
            elif current == group:
                table.setRowHidden(r, fold)
        self._fit_height(table)
        self._recolour()

    def _explain_row(self, row: int) -> None:
        """The full explanation of the row's measure or group, beside it."""
        from .explain_popup import show

        entry = self._rows[row] if 0 <= row < len(self._rows) else None
        table = self.table
        if entry is None or table is None:
            return
        rect = table.visualRect(table.model().index(row, COL_NAME))
        at = table.viewport().mapToGlobal(rect.bottomLeft())
        if isinstance(entry, tuple):
            group = entry[1]
            show(table, f"group.{self._kind}.{group}", at=at,
                 title=dict(GROUPS).get(group, group.capitalize()))
            return
        m = entry
        here = [f"{m.title}: {m.text()}."]
        if m.value is None and m.why_missing:
            here.append(f"Not computed: {m.why_missing}")
        text, level = dataset_text(self.context.get(m.key), m.better)
        if text:
            here.append(f"In this dataset: {text}" + (", on the worse side." if level else "."))
        show(table, f"{self._kind}.{m.key}", at=at, title=m.title, here=" ".join(here),
             fallback=m.help)

    def _fit_columns(self) -> None:
        """The measure's name whole when there is room (shortened with an
        ellipsis when not), then the value, its place in the dataset and the
        info icon. The table fills the panel's width, but not beyond a little
        more than its names need: a wide panel keeps each value near its
        name."""
        table = self.table
        if table is None or not self._needs:
            return
        try:
            room = table.viewport().width()
        except RuntimeError:                # replaced since
            return
        needs = self._needs
        others = [c for c in (COL_VALUE, COL_DATASET, COL_INFO)
                  if not (c == COL_DATASET and not self.context)]
        fixed = sum(needs[c] for c in others)
        for c in others:
            table.setColumnWidth(c, needs[c])
        widest = needs[COL_NAME] + fonts.px(160)
        table.setColumnWidth(COL_NAME, max(fonts.px(90), min(widest, room - fixed)))
        table.setMaximumWidth(widest + fixed + 2 * table.frameWidth() + 2)

    _needs: list = []
    _kind = "anat"

    def _recolour(self, *_a) -> None:
        table = getattr(self, "table", None)
        if table is None:
            return
        from ... import icons

        try:
            theme = ThemeHub.instance().theme
            dim, text = QColor(theme.dim), QColor(theme.text)
            warn = QColor(theme.token("warning", "#d29922"))
            band = QColor(theme.token("surface3", "#2a2f36"))
            info_icon = icons.icon("info")
            for r, m in enumerate(self._rows):
                info = table.item(r, COL_INFO)
                if info is not None:
                    info.setIcon(info_icon)
                if isinstance(m, tuple):
                    head = table.item(r, COL_NAME)
                    if head is not None:
                        head.setIcon(icons.icon("chevron_right" if m[1] in self._collapsed
                                                else "chevron_down"))
                        head.setForeground(text)
                    for c in range(4):
                        item = table.item(r, c)
                        if item is not None:
                            item.setBackground(band)
                    continue
                for c in range(3):
                    item = table.item(r, c)
                    if item is not None:
                        item.setForeground(dim if m.value is None and c else text)
                _t, level = dataset_text(self.context.get(m.key), m.better)
                if level and table.item(r, COL_DATASET) is not None:
                    table.item(r, COL_DATASET).setForeground(warn)
        except RuntimeError:                # the table was replaced
            pass


__all__ = ["FLAG_Z", "GROUPS", "TISSUE_COLOURS", "QualityPanel", "dataset_text", "engine_text",
           "evidence_items", "label_colours", "split_key", "summary_text"]

"""Editor centre pane for a diffusion series' gradient table: its ``.bval``
and ``.bvec`` (either file opens the pair).

What a reviewer looks at, in three cards under a row of facts:

* the facts: volumes, b=0 volumes, shells, whether the table matches the
  image, and the problems the quality check's table checks find;
* **Directions**: each shell's directions on the sphere, folded onto the
  half facing the viewer (a direction and its opposite are one measurement)
  and drawn with an equal-area projection, so an even scheme looks even.
  Seen along x, y or z;
* **b-value of each volume**: the order of the shells and the b=0 volumes;
* **Volumes**: every volume's b-value and direction.

The three are linked: pointing at a volume reads it out above its plot,
clicking it (in either plot or the table) marks it in all three. A shell's
box in the facts row hides or shows it in both plots. Colours follow the
shell in every view (``VizTheme.series`` in the shells' order, b=0 in the
dim colour), and every item is made once and re-coloured on a theme or font
size change (``ThemeHub``), never removed (CLAUDE.md guard 8d).

Read only: a gradient table is the scanner's record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...qc.config import DEFAULT
from .primitives import ElidedLabel, PaneHeader
from .splitter import Splitter

#: Viewing axis -> (horizontal axis, vertical axis, the axis looked along).
VIEWS = {"z": (0, 1, 2), "y": (0, 2, 1), "x": (1, 2, 0)}
_AXIS = "xyz"
#: Rings drawn at these angles from the viewing axis (degrees).
RINGS = (30, 60)


# ---------------------------------------------------------------------------
# Qt-free
# ---------------------------------------------------------------------------


def read_table(path: Path, *, b0_max: float = DEFAULT.dwi.b0_max) -> dict:
    """The gradient table beside ``path`` (a .bval, a .bvec or the image):
    ``bvals``, ``bvecs`` (n x 3), ``shells``, ``n_image`` (the image's
    volumes, or None), ``problems`` and the quality check's ``findings``."""
    from ...qc import dwi
    from ...qc.series import read_gradients
    from ...viz.bids import stem_of

    path = Path(path)
    stem = stem_of(path)
    bvals, bvecs, problems = read_gradients(path.with_name(stem + ".nii.gz"))
    image = next((path.with_name(stem + ext) for ext in (".nii.gz", ".nii")
                  if path.with_name(stem + ext).is_file()), None)
    n_image = None
    if image is not None:
        try:
            import nibabel as nib

            shape = nib.load(str(image)).shape
            n_image = int(shape[3]) if len(shape) > 3 else 1
        except Exception:  # noqa: BLE001 - the facts say what could not be read
            problems = list(problems) + [f"{image.name} could not be read"]
    n = n_image if n_image is not None else (len(bvals) if bvals is not None else 0)
    findings, usable = dwi.check_table(n, bvals, bvecs, list(problems), b0_max=b0_max)
    vecs = None
    if bvecs is not None:
        b = np.asarray(bvecs, dtype=float)
        vecs = b.T if b.shape[0] == 3 and b.ndim == 2 else b
    vals = None if bvals is None else np.asarray(bvals, dtype=float)
    shells = None
    if vals is not None and vecs is not None:
        m = min(len(vals), len(vecs))
        vals, vecs = vals[:m], vecs[:m]
        shells = dwi.shells_of(vals, b0_max)
    return {"bvals": vals, "bvecs": vecs, "shells": shells, "n_image": n_image,
            "image": image, "problems": list(problems), "findings": findings,
            "usable": usable, "stem": stem}


def shell_rows(table: dict) -> list[tuple[int, int, float]]:
    """``(b, directions, largest gap in degrees)`` for each shell above b=0."""
    from ...qc import dwi

    if table.get("shells") is None:
        return []
    shells, vecs = table["shells"], table["bvecs"]
    rows = []
    for s in sorted(int(x) for x in set(shells.tolist()) if x > 0):
        sel = shells == s
        rows.append((s, int(np.count_nonzero(sel)), float(dwi.coverage_gap_deg(vecs[sel]))))
    return rows


def match_text(table: dict) -> Optional[tuple[str, bool]]:
    """Whether the table matches its image: ``(text, matches)``, or None
    when there is no image beside it."""
    if table.get("n_image") is None or table.get("shells") is None:
        return None
    n = len(table["shells"])
    if table["n_image"] == n:
        return f"Matches {table['image'].name} ({n} volumes)", True
    return f"{table['image'].name} has {table['n_image']} volumes, the table {n}", False


def summary_lines(table: dict) -> list[str]:
    """The facts as sentences (the pane shows them as a row; the info
    popup and tests read them here)."""
    if table.get("shells") is None:
        return [*table["problems"]] or ["No gradient table."]
    shells = table["shells"]
    parts = [f"{len(shells)} volumes", f"{int(np.count_nonzero(shells == 0))} at b=0"]
    for s, count, gap in shell_rows(table):
        parts.append(f"{count} at b={s} (largest gap {gap:.0f} degrees)")
    lines = [", ".join(parts) + "."]
    match = match_text(table)
    if match:
        lines.append(match[0] + ".")
    for f in table["findings"]:
        lines.append(f"{f.title}: {f.message}")
    return lines


def hemisphere(vecs: np.ndarray, view: str = "z") -> np.ndarray:
    """Directions (n x 3) as points in the unit disc (n x 2), seen along
    ``view``: folded onto the half of the sphere facing the viewer and drawn
    with Lambert's equal-area projection, so an even spread looks even. The
    centre is the viewing axis, the unit circle the plane across it. A zero
    vector (a b=0 volume) gives NaN."""
    u, v, w = VIEWS[view]
    d = np.asarray(vecs, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    out = np.full((len(d), 2), np.nan)
    ok = norm > 0.5
    unit = d[ok] / norm[ok][:, None]
    unit[unit[:, w] < 0] *= -1.0
    f = 1.0 / np.sqrt(1.0 + unit[:, w])
    out[ok, 0] = unit[:, u] * f
    out[ok, 1] = unit[:, v] * f
    return out


def ring_radius(degrees: float) -> float:
    """The radius, in the disc, of the circle ``degrees`` from the axis."""
    return float(np.sqrt(1.0 - np.cos(np.radians(degrees))))


def _signed(value: float, places: int) -> str:
    """``+0.386`` / ``−0.811``: a true minus sign, so columns line up."""
    return f"{value:+.{places}f}".replace("-", "−")


def describe(table: dict, volume: int, *, direction: bool = True) -> str:
    """One volume, as the readouts say it: its direction above the
    directions, its b-value above the b-values."""
    if not direction:
        return f"Volume {volume}: b={table['bvals'][volume]:g}"
    if int(table["shells"][volume]) == 0:
        return f"Volume {volume}: no direction (b=0)"
    x, y, z = (_signed(c, 3) for c in table["bvecs"][volume])
    return f"Volume {volume} ({x}, {y}, {z})"


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def _qcolor(value: str, alpha: Optional[int] = None) -> QColor:
    from ...viz.theme import parse_colour

    c = QColor(*parse_colour(value))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def dot_icon(colour: str, side: int) -> QIcon:
    """A round dot of ``colour``: a legend mark (a square beside a checkbox
    reads as a second checkbox, rules 4.11)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    ratio = app.devicePixelRatio() if app is not None else 1.0
    pm = QPixmap(max(1, int(side * ratio)), max(1, int(side * ratio)))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_qcolor(colour))
    radius = side * 0.32
    p.drawEllipse(QPointF(side / 2, side / 2), radius, radius)
    p.end()
    return QIcon(pm)


class _Card(QFrame):
    """A rounded card: a title, its own controls, a readout and an info icon
    in a row above what it holds (rules 4.4)."""

    def __init__(self, title: str, explain_key: str, tip: str) -> None:
        super().__init__()
        from .flow_layout import FlowBar

        self.setObjectName("viz-track")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._title_text = title
        self._key = explain_key
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 8, 8)
        v.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("viz-track-title")
        self.title.setToolTip(tip)
        head.addWidget(self.title)
        self.controls = FlowBar(h_spacing=4, v_spacing=4)
        head.addWidget(self.controls)
        self.readout = ElidedLabel("")
        self.readout.setObjectName("viz-track-readout")
        head.addWidget(self.readout, 1)
        self.info = QPushButton()
        self.info.setObjectName("viz-track-btn")
        self.info.setToolTip(tip + " The info icon explains it in full.")
        self.info.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.info.setCursor(Qt.CursorShape.PointingHandCursor)
        self.info.clicked.connect(self._explain)
        head.addWidget(self.info, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(head)
        self.body = v

    def _explain(self) -> None:
        from ..viz.panels import explain_popup

        explain_popup.show(self.info, self._key, title=self._title_text)

    def apply_theme(self) -> None:
        from .. import icons
        from ..viz import fonts

        side = fonts.px(14)
        self.info.setIcon(icons.icon("info"))
        self.info.setIconSize(QSize(side, side))
        self.info.setFixedSize(fonts.px(22), fonts.px(20))


class _Plot:
    """A pyqtgraph plot in a card, its items made once."""

    def __init__(self, card: _Card, *, axes: bool) -> None:
        import pyqtgraph as pg

        self.pg = pg
        self.widget = pg.PlotWidget()
        self.widget.setObjectName("gradient-plot")
        self.item = pi = self.widget.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.setMouseEnabled(x=False, y=False)
        self.vb = pi.getViewBox()
        self.vb.disableAutoRange()
        if axes:
            for name in ("left", "bottom"):
                pi.getAxis(name).enableAutoSIPrefix(False)
        else:
            pi.hideAxis("left")
            pi.hideAxis("bottom")
        pi.layout.setContentsMargins(0, 2, 6, 0)
        self.widget.setMinimumHeight(60)
        card.body.addWidget(self.widget, 1)
        self.scatters: list = []
        # The mark of the volume pointed at, and of the one chosen: a ring
        # over a wider halo in the background colour, so it is found in
        # either theme (rules 4.6).
        self.hover = pg.ScatterPlotItem(pxMode=True)
        self.halo = pg.ScatterPlotItem(pxMode=True)
        self.ring = pg.ScatterPlotItem(pxMode=True)
        for item, z in ((self.hover, 20), (self.halo, 21), (self.ring, 22)):
            item.setZValue(z)
            pi.addItem(item)

    def scatter(self, k: int):
        """The ``k``-th pooled scatter (made on first use, then reused)."""
        while len(self.scatters) <= k:
            s = self.pg.ScatterPlotItem(pxMode=True)
            s.setZValue(10)
            self.item.addItem(s)
            self.scatters.append(s)
        return self.scatters[k]

    def mark(self, item, xy: Optional[tuple[float, float]], size: float, pen) -> None:
        if xy is None or not all(np.isfinite(xy)):
            item.setData([], [])
            return
        item.setData([xy[0]], [xy[1]], size=size, pen=pen, brush=None, symbol="o")

    def contains(self, scene_pos) -> bool:
        return scene_pos is not None and self.vb.sceneBoundingRect().contains(scene_pos)


class GradientPane(QWidget):
    """The gradient table of a diffusion series."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ..viz.panels.panel_header import PanelHeader, corner_button

        self.setObjectName("pane")
        self._path: Optional[Path] = None
        self._table_data: Optional[dict] = None
        self._view = "z"
        self._hidden: set[int] = set()
        self._selected: Optional[int] = None
        self._hovered: Optional[int] = None
        self._shell_boxes: dict[int, QCheckBox] = {}
        self._colours: dict[int, str] = {}
        self._sphere_xy: Optional[np.ndarray] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._header = PaneHeader("Gradient table")
        v.addWidget(self._header)

        # The facts, the shells' boxes, and what the table is (corner).
        self._facts_bar = PanelHeader(margins=(12, 6, 8, 6))
        v.addWidget(self._facts_bar)
        self._fact_labels: list[QLabel] = []
        self._match_chip = None
        info = corner_button("info", "What the gradient table is, and how it is checked")
        info.clicked.connect(lambda: self._explain_table(info))
        self._facts_bar.add_corner(info)

        # Problems the table checks find, one card each.
        self._findings = QWidget()
        self._findings.setObjectName("viz-tracks")
        self._findings_layout = QVBoxLayout(self._findings)
        self._findings_layout.setContentsMargins(8, 8, 8, 0)
        self._findings_layout.setSpacing(6)
        self._findings.setVisible(False)
        v.addWidget(self._findings)

        # Directions and b-values in a column, the volumes beside them.
        self._split = Splitter(Qt.Orientation.Horizontal)
        self._split.setObjectName("viz-tracks")
        self._split.setChildrenCollapsible(False)
        self._split_moved = False
        self._split.splitterMoved.connect(lambda *_: setattr(self, "_split_moved", True))
        plots = Splitter(Qt.Orientation.Vertical)
        plots.setObjectName("viz-tracks")
        plots.setChildrenCollapsible(False)
        self._dir_card = _Card(
            "Directions", "plot.dwi.directions",
            "Each shell's gradient directions on the sphere, seen along one axis, with "
            "the half facing away folded over (a direction and its opposite are one "
            "measurement).")
        self.sphere = _Plot(self._dir_card, axes=False)
        self.sphere.item.setAspectLocked(True)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        self._view_buttons: dict[str, QPushButton] = {}
        along = QLabel("Along")
        along.setObjectName("viz-track-readout")
        along.setToolTip("The image axis the sphere is seen along")
        self._dir_card.controls.addWidget(along)
        for axis in ("x", "y", "z"):
            b = QPushButton(axis)
            b.setObjectName("tb-btn-toggle")
            b.setCheckable(True)
            b.setToolTip(f"See the sphere looking along the image's {axis} axis")
            b.setChecked(axis == self._view)
            b.clicked.connect(lambda _c=False, a=axis: self.set_view(a))
            self._view_group.addButton(b)
            self._dir_card.controls.addWidget(b)
            self._view_buttons[axis] = b
        pg = self.sphere.pg
        self._guides = pg.PlotCurveItem(connect="finite")
        self._cross = pg.PlotCurveItem(connect="pairs")
        for item in (self._guides, self._cross):
            item.setZValue(1)
            self.sphere.item.addItem(item)
        self._axis_labels = []
        for anchor in ((1, 0.5), (0, 0.5), (0.5, 1), (0.5, 0)):
            t = pg.TextItem("", anchor=anchor)
            t.setZValue(2)
            self.sphere.item.addItem(t)
            self._axis_labels.append(t)

        self._b_card = _Card(
            "b-value of each volume", "plot.dwi.bvalues",
            "Every volume's b-value in the order acquired, one colour per shell and b=0 "
            "in grey.")
        self.bplot = _Plot(self._b_card, axes=True)
        plots.addWidget(self._dir_card)
        plots.addWidget(self._b_card)
        plots.setStretchFactor(0, 3)
        plots.setStretchFactor(1, 2)
        plots_box = QWidget()
        plots_box.setObjectName("viz-tracks")
        pl = QVBoxLayout(plots_box)
        pl.setContentsMargins(8, 8, 4, 8)
        pl.addWidget(plots)
        self._split.addWidget(plots_box)

        self._table_card = _Card(
            "Volumes", "group.dwi.gradients",
            "Every volume's b-value (after a dot in its shell's colour) and direction, as "
            "the .bval and .bvec give them.")
        self.table = QTableWidget(0, 5)
        self.table.setObjectName("gradient-table")
        self.table.setHorizontalHeaderLabels(["Volume", "b", "x", "y", "z"])
        self.table.horizontalHeaderItem(1).setToolTip(
            "The b-value in s/mm², after a dot in its shell's colour")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Every column to its contents: a stretched last column keeps Qt's
        # default 100 px when space is short (rules 4.16).
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self._table_card.body.addWidget(self.table, 1)
        table_box = QWidget()
        table_box.setObjectName("viz-tracks")
        tl = QVBoxLayout(table_box)
        tl.setContentsMargins(4, 8, 8, 8)
        tl.addWidget(self._table_card)
        self._split.addWidget(table_box)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 0)
        v.addWidget(self._split, 1)

        self.sphere.vb.sigResized.connect(lambda *_: self._on_sphere_resized())
        for plot, moved in ((self.sphere, self._on_sphere_moved),
                            (self.bplot, self._on_bplot_moved)):
            scene = plot.widget.scene()
            scene.sigMouseMoved.connect(moved)
            scene.sigMouseClicked.connect(lambda e, p=plot: self._on_clicked(p, e))
            plot.widget.leaveEvent = lambda _e: self._set_hover(None)

        from ..viz.bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w._apply_theme())
        self._apply_theme()

    # -- binding ---------------------------------------------------------------

    def set_read_only(self, on: bool) -> None:
        """Always read only; here for the Editor's mode switch."""
        del on

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        del root
        self._path = Path(path) if path is not None else None
        self._table_data = (read_table(self._path, b0_max=self._b0_max())
                            if self._path is not None else None)
        self._selected = self._hovered = None
        self._hidden = set()
        if self._table_data is not None:
            self._header.setText(f"{self._table_data['stem']} gradient table")
        self._fill_facts()
        self._fill_findings()
        self._fill_table()
        self._apply_theme()
        self._fit_split()

    def current_file(self) -> Optional[Path]:
        return self._path

    @staticmethod
    def _b0_max() -> float:
        """The b=0 limit the quality check uses (Settings > Quality control)."""
        try:
            from ...viz.settings import qc_config
            from ..viz.bridge import SettingsHub

            return float(qc_config(SettingsHub.instance().settings).dwi.b0_max)
        except Exception:  # noqa: BLE001 - the default is always usable
            return float(DEFAULT.dwi.b0_max)

    def facts(self) -> list[str]:
        """What the facts row says, in order."""
        out = [lab.text() for lab in self._fact_labels]
        if self._match_chip is not None:
            out.append(self._match_chip.text())
        return out

    def view(self) -> str:
        return self._view

    def set_view(self, axis: str) -> None:
        """Look at the sphere along ``axis`` ("x", "y" or "z")."""
        if axis not in VIEWS:
            return
        self._view = axis
        self._view_buttons[axis].setChecked(True)
        self._draw()

    def set_shell_shown(self, shell: int, shown: bool) -> None:
        (self._hidden.discard if shown else self._hidden.add)(int(shell))
        box = self._shell_boxes.get(int(shell))
        if box is not None and box.isChecked() != shown:
            box.setChecked(shown)
        self._draw()

    def selected_volume(self) -> Optional[int]:
        return self._selected

    def select_volume(self, volume: Optional[int]) -> None:
        """Mark ``volume`` in both plots and the table."""
        t = self._table_data
        if volume is not None and (t is None or t["shells"] is None
                                   or not 0 <= volume < len(t["shells"])):
            volume = None
        self._selected = volume
        if volume is not None and self.table.currentRow() != volume:
            self.table.blockSignals(True)
            self.table.selectRow(volume)
            self.table.blockSignals(False)
            item = self.table.item(volume, 0)
            if item is not None:
                self.table.scrollToItem(item)
        self._draw_marks()
        self._show_readouts()

    # -- facts and findings ----------------------------------------------------

    def _fill_facts(self) -> None:
        from ..viz.panels import explain_popup
        from .primitives import Chip

        bar = self._facts_bar.controls
        for w in list(bar.widgets()):
            bar.removeWidget(w)
            w.deleteLater()
        self._fact_labels, self._shell_boxes, self._match_chip = [], {}, None
        t = self._table_data
        if t is None:
            return
        if t["shells"] is None:
            lab = QLabel("No usable gradient table")
            lab.setObjectName("viz-track-title")
            bar.addWidget(lab)
            self._fact_labels.append(lab)
            return
        shells = t["shells"]
        for text in (f"{len(shells)} volumes", f"{int(np.count_nonzero(shells == 0))} at b=0",
                     f"{len(shell_rows(t))} shells"):
            lab = QLabel(text)
            lab.setObjectName("viz-track-title")
            bar.addWidget(lab)
            self._fact_labels.append(lab)
        match = match_text(t)
        if match is not None:
            chip = Chip(match[0], "success" if match[1] else "err")
            chip.setToolTip("The number of volumes in the image beside the table")
            bar.addWidget(chip)
            self._match_chip = chip
        gap_tip = explain_popup.hover_text("dwi.coverage_gap_b1000")
        for s, count, gap in shell_rows(t):
            box = QCheckBox(f"b={s} · {count} directions · gap {gap:.0f}°")
            box.setChecked(True)
            box.setToolTip(f"Show the b={s} shell in both plots. Largest gap between its "
                           f"directions: {gap:.0f} degrees. {gap_tip}")
            box.toggled.connect(lambda on, s=s: self.set_shell_shown(s, on))
            bar.addWidget(box)
            self._shell_boxes[s] = box

    def _fill_findings(self) -> None:
        from .primitives import Chip

        lay = self._findings_layout
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        t = self._table_data
        levels = {"error": (0, "Error", "err"), "warning": (1, "Warning", "warn"),
                  "info": (2, "Note", "accent")}
        found = sorted((f for f in (t["findings"] if t else []) if f.level in levels),
                       key=lambda f: levels[f.level][0])
        for f in found:
            card = QFrame()
            card.setObjectName("viz-track")
            card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row = QHBoxLayout(card)
            row.setContentsMargins(10, 6, 8, 6)
            row.setSpacing(8)
            _o, text, kind = levels[f.level]
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
            lay.addWidget(card)
        self._findings.setVisible(bool(found))

    def _explain_table(self, anchor: QWidget) -> None:
        from ..viz.panels import explain_popup

        here = " ".join(summary_lines(self._table_data)) if self._table_data else ""
        explain_popup.show(anchor, "group.dwi.gradients", here=here)

    # -- the table -------------------------------------------------------------

    def _fill_table(self) -> None:
        t = self._table_data
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if t is not None and t["shells"] is not None:
            bvals, vecs, shells = t["bvals"], t["bvecs"], t["shells"]
            n = len(shells)
            self.table.setRowCount(n)
            right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            for i in range(n):
                cells = [str(i), f"{bvals[i]:g}", *(_signed(c, 3) for c in vecs[i])]
                shell = int(shells[i])
                for c, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if c:
                        item.setTextAlignment(right)
                    if c == 1:
                        item.setToolTip("b=0" if shell == 0 else f"Shell b={shell}")
                    self.table.setItem(i, c, item)
        self.table.blockSignals(False)
        self._table_card.readout.setText(
            "" if t is None or t["shells"] is None else
            "Click a row to mark it in the plots")

    def _paint_table_marks(self) -> None:
        """Each row's shell as a dot in its colour (the plots' colours),
        before its b-value."""
        from ..viz import fonts

        t = self._table_data
        if t is None or t["shells"] is None:
            return
        side = fonts.px(12)
        dots = {s: dot_icon(c, side) for s, c in self._colours.items()}
        for i, s in enumerate(t["shells"]):
            item = self.table.item(i, 1)
            if item is not None:
                item.setIcon(dots.get(int(s), QIcon()))
        self.table.setIconSize(QSize(side, side))

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        self.select_volume(rows[0].row() if rows else None)

    def _table_width(self) -> int:
        """The width that shows every column of the table, in its card."""
        t = self.table
        t.ensurePolished()
        header = t.horizontalHeader()
        width = sum(max(t.sizeHintForColumn(c), header.sectionSizeHint(c))
                    for c in range(t.columnCount())) + 2 * t.frameWidth()
        if t.rowCount():
            width += t.verticalScrollBar().sizeHint().width()
        m = self._table_card.body.contentsMargins()
        # The card's own margins and border, and the column's.
        return width + m.left() + m.right() + 2 + 12

    def _fit_split(self) -> None:
        from ..viz import fonts

        total = self._split.width()
        if self._split_moved or total <= 0:
            return
        # Every column if the plots keep a usable width, else half each.
        table = min(self._table_width(), max(total - fonts.px(300), total // 2, 160))
        self._split.setSizes([total - table, table])

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_split()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Filled while hidden, the table was measured before its style
        # applied, and a page of a stack is sized before it is shown.
        super().showEvent(event)
        self._fit_split()

    # -- look ------------------------------------------------------------------

    def _shell_colours(self) -> dict[int, str]:
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        colours = {0: theme.dim}
        if self._table_data is not None:
            for k, (s, _n, _g) in enumerate(shell_rows(self._table_data)):
                colours[s] = theme.series(k)
        return colours

    def _apply_theme(self) -> None:
        """Every colour and size from the theme and the font setting."""
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        pg = self.sphere.pg
        self._colours = self._shell_colours()
        fg = _qcolor(theme.plot_foreground, 200).name()
        for plot in (self.sphere, self.bplot):
            plot.widget.setBackground(theme.plot_background)
        pi = self.bplot.item
        for name in ("left", "bottom"):
            pi.getAxis(name).setPen(_qcolor(theme.plot_foreground, 120))
        fonts.style_axes(pi, fg)
        fonts.axis_title(pi, "bottom", "Volume", fg)
        fonts.axis_title(pi, "left", "b (s/mm²)", fg)
        pi.getAxis("left").setWidth(fonts.axis_width() + fonts.px(16))
        self._guides.setPen(pg.mkPen(_qcolor(theme.grid), width=1))
        self._cross.setPen(pg.mkPen(_qcolor(theme.grid), width=1,
                                    style=Qt.PenStyle.DashLine))
        for label in self._axis_labels:
            label.setColor(_qcolor(theme.dim))
            label.setFont(fonts.font(fonts.LABEL_PX))
        for card in (self._dir_card, self._b_card, self._table_card):
            card.apply_theme()
        side = fonts.px(12)
        for s, box in self._shell_boxes.items():
            box.setIcon(dot_icon(self._colours.get(s, theme.dim), side))
            box.setIconSize(QSize(side, side))
        self._paint_table_marks()
        self._draw()

    def repaint_for_palette(self, pal: dict) -> None:
        """QSS-styled children re-polished (the table kept the old theme's
        colours without it); the plots follow the theme hub."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()

    # -- drawing ---------------------------------------------------------------

    def _draw(self) -> None:
        self._draw_sphere()
        self._draw_bvalues()
        self._draw_marks()
        self._show_readouts()

    def _draw_sphere(self) -> None:
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        pg = self.sphere.pg
        # The circle across the viewing axis, two rings, and a cross.
        angle = np.linspace(0.0, 2 * np.pi, 241)
        xs: list[float] = []
        ys: list[float] = []
        for r in (1.0, *(ring_radius(a) for a in RINGS)):
            xs += [*(r * np.cos(angle)), np.nan]
            ys += [*(r * np.sin(angle)), np.nan]
        self._guides.setData(np.array(xs), np.array(ys))
        self._cross.setData(np.array([-1.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0, 1.0]))
        u, v, _w = VIEWS[self._view]
        gap = 1.05
        minus = "−"
        for label, (text, x, y) in zip(self._axis_labels, (
                (f"{minus}{_AXIS[u]}", -gap, 0.0), (f"+{_AXIS[u]}", gap, 0.0),
                (f"+{_AXIS[v]}", 0.0, gap), (f"{minus}{_AXIS[v]}", 0.0, -gap))):
            label.setText(text)
            label.setPos(x, y)
        t = self._table_data
        size = self._dot_px()
        edge = pg.mkPen(_qcolor(theme.plot_background), width=1.2)
        used = 0
        self._sphere_xy = None
        if t is not None and t["shells"] is not None:
            xy = hemisphere(t["bvecs"], self._view)
            xy[t["shells"] == 0] = np.nan
            self._sphere_xy = xy
            for s, _n, _g in shell_rows(t):
                sel = t["shells"] == s
                item = self.sphere.scatter(used)
                item.setData(xy[sel, 0], xy[sel, 1], size=size, pen=edge,
                             brush=pg.mkBrush(_qcolor(self._colours.get(s, theme.dim))),
                             symbol="o")
                item.setVisible(s not in self._hidden)
                used += 1
        for item in self.sphere.scatters[used:]:
            item.setData([], [])
        self._fit_sphere()

    def _short_side(self) -> float:
        vb = self.sphere.vb
        return max(1.0, min(vb.width(), vb.height()))

    def _dot_px(self) -> int:
        """The directions' dots: their size at the font setting, smaller in a
        small plot so they do not pile up."""
        from ..viz import fonts

        return int(max(fonts.px(5), min(fonts.px(9), self._short_side() / 34)))

    def _fit_sphere(self) -> None:
        """The disc whole and its axis letters beside it, at any size: the
        range leaves a letter's height around the circle."""
        from ..viz import fonts

        letter = fonts.px(fonts.LABEL_PX) * 1.7
        room = min(2.2, 1.05 / max(1.0 - 2.0 * letter / self._short_side(), 0.48))
        self.sphere.vb.setRange(xRange=(-room, room), yRange=(-room, room), padding=0)

    def _on_sphere_resized(self) -> None:
        size = self._dot_px()
        for item in self.sphere.scatters:
            if len(item.data):
                item.setSize(size)
        self._fit_sphere()
        self._draw_marks()

    def _draw_bvalues(self) -> None:
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        pg = self.bplot.pg
        t = self._table_data
        size = fonts.px(7)
        edge = pg.mkPen(_qcolor(theme.plot_background), width=1.0)
        used = 0
        axis = self.bplot.item.getAxis("left")
        if t is not None and t["shells"] is not None:
            bvals, shells = t["bvals"], t["shells"]
            index = np.arange(len(bvals), dtype=float)
            rows = shell_rows(t)
            for s in [0, *(r[0] for r in rows)]:
                sel = shells == s
                if not np.any(sel):
                    continue
                item = self.bplot.scatter(used)
                item.setData(index[sel], bvals[sel], size=size, pen=edge,
                             brush=pg.mkBrush(_qcolor(self._colours.get(s, theme.dim))),
                             symbol="o")
                item.setVisible(s == 0 or s not in self._hidden)
                used += 1
            top = float(np.nanmax(bvals)) if len(bvals) else 1.0
            levels = sorted({0, *(r[0] for r in rows)})
            axis.setTicks([[(float(b), f"{b:g}") for b in levels], []])
            self.bplot.vb.setRange(xRange=(-0.5, max(len(bvals) - 0.5, 1.0)),
                                   yRange=(-0.08 * max(top, 1.0), 1.1 * max(top, 1.0)),
                                   padding=0)
            self.bplot.item.showGrid(x=False, y=True, alpha=0.2)
        else:
            axis.setTicks(None)
        for item in self.bplot.scatters[used:]:
            item.setData([], [])

    def _point(self, plot: _Plot, volume: Optional[int]) -> Optional[tuple[float, float]]:
        t = self._table_data
        if volume is None or t is None or t["shells"] is None:
            return None
        shell = int(t["shells"][volume])
        if shell in self._hidden:
            return None
        if plot is self.sphere:
            if self._sphere_xy is None:
                return None
            x, y = self._sphere_xy[volume]
            return (float(x), float(y)) if np.isfinite(x) else None
        return float(volume), float(t["bvals"][volume])

    def _draw_marks(self) -> None:
        from ..viz import fonts
        from ..viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        pg = self.sphere.pg
        for plot, size in ((self.sphere, self._dot_px()), (self.bplot, fonts.px(7))):
            plot.mark(plot.hover, self._point(plot, self._hovered), size + fonts.px(6),
                      pg.mkPen(_qcolor(theme.plot_foreground), width=1.5))
            chosen = self._point(plot, self._selected)
            plot.mark(plot.halo, chosen, size + fonts.px(8),
                      pg.mkPen(_qcolor(theme.plot_background), width=5))
            plot.mark(plot.ring, chosen, size + fonts.px(8),
                      pg.mkPen(_qcolor(theme.token("warning", "#d29922")), width=2))

    def _show_readouts(self) -> None:
        t = self._table_data
        volume = self._hovered if self._hovered is not None else self._selected
        known = t is not None and t["shells"] is not None and volume is not None
        self._dir_card.readout.setText(describe(t, volume) if known else "")
        self._b_card.readout.setText(describe(t, volume, direction=False) if known else "")

    # -- pointing --------------------------------------------------------------

    def _nearest(self, plot: _Plot, scene_pos) -> Optional[int]:
        """The volume drawn nearest the pointer in ``plot``, if close."""
        from ..viz import fonts

        t = self._table_data
        if t is None or t["shells"] is None or not plot.contains(scene_pos):
            return None
        at = plot.vb.mapSceneToView(scene_pos)
        sx, sy = plot.vb.viewPixelSize()
        hidden = np.isin(t["shells"], [s for s in self._hidden if s])
        if plot is self.sphere:
            if self._sphere_xy is None:
                return None
            xy = self._sphere_xy.copy()
        else:
            xy = np.column_stack([np.arange(len(t["bvals"]), dtype=float), t["bvals"]])
        xy[hidden] = np.nan
        d = np.hypot((xy[:, 0] - at.x()) / max(sx, 1e-12), (xy[:, 1] - at.y()) / max(sy, 1e-12))
        if not np.any(np.isfinite(d)):
            return None
        i = int(np.nanargmin(d))
        return i if d[i] <= fonts.px(10) else None

    def _set_hover(self, volume: Optional[int]) -> None:
        if volume == self._hovered:
            return
        self._hovered = volume
        self._draw_marks()
        self._show_readouts()

    def _on_sphere_moved(self, pos) -> None:
        self._set_hover(self._nearest(self.sphere, pos))

    def _on_bplot_moved(self, pos) -> None:
        self._set_hover(self._nearest(self.bplot, pos))

    def _on_clicked(self, plot: _Plot, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        found = self._nearest(plot, event.scenePos())
        if found is not None:
            self.select_volume(found)


__all__ = ["GradientPane", "VIEWS", "describe", "dot_icon", "hemisphere", "match_text",
           "read_table", "ring_radius", "shell_rows", "summary_lines"]

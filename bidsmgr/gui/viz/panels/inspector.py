"""The controls column: every setting of the volume viewer, by purpose.

One column of collapsible sections replaces the toolbar's second row, the
Layers panel and the 3-D panel, which between them mixed view modes,
contrast, crosshair style and twenty-five unlabelled sliders. The sections:

* **Layers**: the images drawn, top-most first, each with its eye.
* **Display**: how the selected layer's values become colours (colour map,
  window over its histogram, gamma, opacity, tails).
* **Overlay**: what only a layer drawn over another needs.
* **View**: the conventions every slice shares, and the crosshair.
* **Layout**: how the views are arranged (``Scene.layout``).
* **3-D**: one section per purpose (look, edges, peeling, lighting,
  overlays in 3-D, quality) and the clipping planes, shown while a 3-D view
  is on screen, each showing only what the current effect uses.

Every number is a :class:`~..controls.NumberControl` (slider + value + unit)
or a :class:`~..controls.RangeControl`, every control RUNS A COMMAND (so it
is undoable, linkable and scriptable), and a drag is one undo step. Sections
remember whether they are open. Generated from :data:`bidsmgr.viz.props.LAYER_PROPS`
and :data:`bidsmgr.viz.render3d.PARAMS`: a property added there gets its
control here without widget code.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QImage, QPainter, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ....viz import actions as A
from ....viz import props as P
from ....viz import render3d, views
from ....viz.compute import colormaps
from ....viz.scene import ClipPlane
from ..context import ViewerContext
from ..controls import NumberControl, RangeControl

#: The clip-plane arrangements ``clip.preset`` knows, as the panel lists them.
CLIP_PRESETS = (("none", "No cut"), ("one", "Half, at the crosshair"),
                ("wedge", "Wedge, at the crosshair"), ("corner", "Corner, at the crosshair"),
                ("four", "Crop four sides"), ("box", "Crop to a box"))

_SWATCHES: dict[str, QIcon] = {}


def swatch_icon(name: str) -> QIcon:
    """A colour map as a small strip, for the colour map lists."""
    icon = _SWATCHES.get(name)
    if icon is None:
        strip = np.ascontiguousarray(np.repeat(colormaps.swatch(name, 64), 10, axis=0))
        img = QImage(strip.data, 64, 10, 64 * 4, QImage.Format.Format_RGBA8888).copy()
        icon = QIcon(QPixmap.fromImage(img))
        _SWATCHES[name] = icon
    return icon


# ---------------------------------------------------------------------------
# A section
# ---------------------------------------------------------------------------


class _Header(QWidget):
    """A section's title bar: caret and title, painted; click to fold. A
    section with defaults has a restore button at the right."""

    clicked = pyqtSignal()
    #: The restore button was clicked.
    reset_clicked = pyqtSignal()
    RESET_PX = 22

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.open = True
        #: Whether the restore button is shown.
        self.resettable = False
        self._fit_height()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        from ..bridge import ThemeHub, connect_while_alive

        # A font-size change re-applies the theme: the bar grows with its title.
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w._fit_height())

    def _fit_height(self) -> None:
        from .. import fonts

        height = max(28, fonts.px(28))
        if self.height() != height:
            self.setFixedHeight(height)

    def sizeHint(self) -> QSize:  # noqa: N802
        from .. import fonts

        return QSize(200, max(28, fonts.px(28)))

    def reset_rect(self) -> QRectF:
        w = self.RESET_PX
        return QRectF(self.width() - w - 6, (self.height() - w) / 2.0, w, w)

    def _on_reset(self, pos) -> bool:
        return self.resettable and self.reset_rect().contains(QPointF(pos))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_reset(event.position()):
                self.reset_clicked.emit()
            else:
                self.clicked.emit()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().mouseMoveEvent(event)

    def event(self, event) -> bool:  # noqa: D401
        if event.type() == QEvent.Type.ToolTip:
            from PyQt6.QtWidgets import QToolTip

            if self._on_reset(QPointF(event.pos())):
                QToolTip.showText(event.globalPos(),
                                  f"Restore the defaults of {self.title.lower()} (undoable)",
                                  self)
            else:
                QToolTip.showText(event.globalPos(),
                                  "Click to fold or unfold" if self.open else
                                  "Click to unfold", self)
            return True
        return super().event(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        from ..bridge import ThemeHub

        theme = ThemeHub.instance().theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.underMouse():
            p.fillRect(self.rect(), QColor(theme.token("surface3", "#1c2128")))
        dim = QColor(theme.dim)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dim)
        cx, cy = 14.0, self.height() / 2.0
        if self.open:
            tri = [QPointF(cx - 4, cy - 2), QPointF(cx + 4, cy - 2), QPointF(cx, cy + 3)]
        else:
            tri = [QPointF(cx - 2, cy - 4), QPointF(cx - 2, cy + 4), QPointF(cx + 3, cy)]
        p.drawPolygon(QPolygonF(tri))
        from .. import fonts

        font = fonts.font(11, bold=True)
        font.setCapitalization(QFont.Capitalization.AllUppercase)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        p.setFont(font)
        p.setPen(QColor(theme.text) if self.open else dim)
        right = self.RESET_PX + 12 if self.resettable else 4
        p.drawText(QRectF(26, 0, self.width() - 26 - right, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.title)
        if self.resettable:
            from ... import icons

            r = self.reset_rect()
            hover = self.underMouse() and r.contains(QPointF(self.mapFromGlobal(QCursor.pos())))
            if hover:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(theme.token("surface2", "#161b22")))
                p.drawRoundedRect(r, 5.0, 5.0)
            colour = theme.text if hover else theme.dim
            pix = icons.icon("restore", colour).pixmap(16, 16)
            p.drawPixmap(int(r.center().x() - 8), int(r.center().y() - 8), pix)
        p.end()


class Section(QWidget):
    """A titled, collapsible group of rows (label left, control right)."""

    def __init__(self, inspector: "Inspector", key: str, title: str, *,
                 open_: bool = True) -> None:
        super().__init__()
        self.inspector = inspector
        self.ctx = inspector.ctx
        self.key = key
        self.setObjectName("viz-section")
        # A plain QWidget paints its stylesheet background only when asked.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 2)
        v.setSpacing(0)
        self.header = _Header(title)
        self.header.clicked.connect(self.toggle)
        self.header.reset_clicked.connect(self.restore_defaults)
        self.header.resettable = type(self).restore_defaults is not Section.restore_defaults
        v.addWidget(self.header)
        self.body = QWidget()
        self.body.setObjectName("viz-section-body")
        self.body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(14, 4, 10, 10)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(7)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnMinimumWidth(0, 92)
        v.addWidget(self.body)
        self._rows: dict[str, tuple[Optional[QLabel], QWidget]] = {}
        saved = self.ctx.settings.inspector_sections.get(key)
        self.set_open(open_ if saved is None else bool(saved), remember=False)

    # -- rows ------------------------------------------------------------
    def add_row(self, key: str, label: Optional[str], widget: QWidget, help_: str = "",
                *, span: bool = False) -> QWidget:
        r = self.grid.rowCount()
        lab = None
        if label is not None and not span:
            lab = QLabel(label)
            lab.setObjectName("sidecar-footer-summary")
            lab.setWordWrap(True)
            lab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            if help_:
                lab.setToolTip(help_)
            self.grid.addWidget(lab, r, 0, Qt.AlignmentFlag.AlignVCenter)
            self.grid.addWidget(widget, r, 1)
        else:
            self.grid.addWidget(widget, r, 0, 1, 2)
        if help_:
            widget.setToolTip(help_)
        self._rows[key] = (lab, widget)
        return widget

    def show_row(self, key: str, on: bool) -> None:
        lab, widget = self._rows[key]
        if widget.isHidden() == on:
            widget.setVisible(on)
        if lab is not None and lab.isHidden() == on:
            lab.setVisible(on)

    def row(self, key: str) -> QWidget:
        return self._rows[key][1]

    def row_shown(self, key: str) -> bool:
        return not self._rows[key][1].isHidden()

    def rows(self) -> list[str]:
        return list(self._rows)

    # -- defaults --------------------------------------------------------
    def restore_defaults(self) -> None:
        """Put this section's settings back as installed (a section with
        none to restore does not override this, and shows no button)."""

    # -- folding ---------------------------------------------------------
    def is_open(self) -> bool:
        return self.header.open

    def set_open(self, on: bool, *, remember: bool = True) -> None:
        self.header.open = bool(on)
        self.body.setVisible(bool(on))
        self.header.update()
        if remember:
            key = self.key
            self.ctx.settings_hub.update(
                lambda s: s.inspector_sections.__setitem__(key, bool(on)))

    def toggle(self) -> None:
        self.set_open(not self.is_open())

    # -- helpers ---------------------------------------------------------
    def number(self, lo: float, hi: float, *, step: float = 0.01, unit: str = "",
               default: Optional[float] = None, log: bool = False,
               on_change: Callable[[float], None]) -> NumberControl:
        nc = NumberControl(lo, hi, step=step, unit=unit, default=default, log=log)
        nc.value_changed.connect(lambda v: self.inspector.run_guarded(on_change, v))
        nc.pressed.connect(self.ctx.store.begin_gesture)
        nc.released.connect(self.ctx.store.end_gesture)
        return nc

    def check(self, text: str, on_change: Callable[[bool], None]) -> QCheckBox:
        box = QCheckBox(text)
        box.toggled.connect(lambda v: self.inspector.run_guarded(on_change, bool(v)))
        return box

    def combo(self, items, on_change: Callable[[Any], None]) -> QComboBox:
        box = _combo()
        for value, text in items:
            box.addItem(text, value)
        box.currentIndexChanged.connect(
            lambda _i, b=box: self.inspector.run_guarded(on_change, b.currentData()))
        return box

    def applies(self) -> bool:
        return True

    def sync(self) -> None:  # pragma: no cover - overridden
        pass


def _combo() -> QComboBox:
    """A combo box that never widens the column: as wide as the column
    gives it, its popup showing the whole text. By default a combo is as
    wide as its longest item, and one long choice cut every control off."""
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(8)
    return box


def _set_combo(box: QComboBox, value) -> None:
    i = box.findData(value)
    if i >= 0 and box.currentIndex() != i:
        box.setCurrentIndex(i)


def _set_check(box: QCheckBox, on: bool) -> None:
    if box.isChecked() != bool(on):
        box.setChecked(bool(on))


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class LayersSection(Section):
    def __init__(self, inspector) -> None:
        super().__init__(inspector, "layers", "Layers")
        self.list = QListWidget()
        self.list.setObjectName("viz-layer-list")
        # A long file name is elided, never allowed to widen the column past
        # the window (it pushed every control off the right edge).
        self.list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.list.currentItemChanged.connect(self._on_select)
        self.list.itemChanged.connect(self._on_item_changed)
        self.add_row("list", None, self.list, span=True)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.add_button = self._button("Add...", "Draw another image over this one",
                                       inspector.add_requested.emit)
        self.remove_button = self._button("Remove", "Take the selected overlay off",
                                          lambda: self._run_selected("layer.remove"))
        self.up_button = self._button("Up", "Draw the selected overlay above the next",
                                      lambda: self._run_selected("layer.move", by=1))
        self.down_button = self._button("Down", "Draw the selected overlay below the previous",
                                        lambda: self._run_selected("layer.move", by=-1))
        for b in (self.add_button, self.remove_button, self.up_button, self.down_button):
            row.addWidget(b)
        row.addStretch(1)
        self.add_row("buttons", None, buttons, span=True)

    def _button(self, text, tip, slot) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("tb-btn")
        b.setToolTip(tip)
        b.clicked.connect(lambda _c=False: slot())
        return b

    def _run_selected(self, command_id: str, **params) -> None:
        try:
            self.ctx.run(command_id, layer=self.inspector.selected_id(), **params)
        except ValueError as exc:
            self.ctx.status.emit(str(exc))

    def _on_select(self, current, _previous) -> None:
        if current is None or self.inspector.syncing:
            return
        self.ctx.select_layer(current.data(Qt.ItemDataRole.UserRole))

    def _on_item_changed(self, item) -> None:
        if self.inspector.syncing:
            return
        self.ctx.run("layer.set", layer=item.data(Qt.ItemDataRole.UserRole),
                     visible=item.checkState() == Qt.CheckState.Checked)

    def sync(self) -> None:
        scene = self.ctx.scene
        ordered = list(reversed(scene.layers))
        ids = [lay.id for lay in ordered]
        if [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())] != ids:
            self.list.clear()
            for lay in ordered:
                item = QListWidgetItem(lay.name or lay.id)
                item.setData(Qt.ItemDataRole.UserRole, lay.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self.list.addItem(item)
        # As tall as its rows (up to six), never an empty box.
        row_h = max(self.list.sizeHintForRow(0), 20) if self.list.count() else 22
        self.list.setFixedHeight(min(max(len(ordered), 1), 6) * row_h + 6)
        selected = self.inspector.selected_id()
        for i, lay in enumerate(ordered):
            item = self.list.item(i)
            src = views.source_of(self.ctx.store, lay)
            tag = f"   {src.n_frames} vol" if src is not None and src.is_4d else ""
            item.setText((lay.name or lay.id) + tag)
            item.setCheckState(Qt.CheckState.Checked if lay.visible else Qt.CheckState.Unchecked)
            if lay.id == selected:
                self.list.setCurrentRow(i)
        base = scene.base_layer()
        overlays = [lay.id for lay in scene.layers if base is None or lay.id != base.id]
        is_overlay = selected in overlays
        self.remove_button.setEnabled(is_overlay)
        self.up_button.setEnabled(is_overlay and overlays.index(selected) < len(overlays) - 1)
        self.down_button.setEnabled(is_overlay and overlays.index(selected) > 0)
        self.add_button.setEnabled(base is not None)


# ---------------------------------------------------------------------------
# The selected layer's look
# ---------------------------------------------------------------------------


class LayerPropsSection(Section):
    """The props of one group of :data:`bidsmgr.viz.props.LAYER_PROPS`."""

    def __init__(self, inspector, key: str, title: str, group: str) -> None:
        super().__init__(inspector, key, title)
        self.props = [p for p in P.LAYER_PROPS if p.group == group and p.key != "visible"]
        self._controls: dict[str, QWidget] = {}
        for prop in self.props:
            widget = self._make(prop)
            self._controls[prop.key] = widget
            self.add_row(prop.key, None if prop.kind == "bool" else prop.title, widget, prop.help)

    def restore_defaults(self) -> None:
        """The selected layer's look as it opened: the user's defaults for
        the base image, the look its content called for for an overlay."""
        layer = self.inspector.selected_layer()
        if layer is None:
            return
        default = self.inspector.presenter.default_look(layer)
        if self.key == "display":
            # The look, not the overlay options (they have their own section).
            default["outline_px"] = layer.display.outline_px
            self.ctx.run("layer.reset", layer=layer.id, display=default)
        else:
            params = {"outline_px": float(default.get("outline_px", 0.0))}
            if self.inspector.is_overlay(layer.id):
                params["in_3d"] = True
            self.ctx.run("layer.set", layer=layer.id, **params)

    def _set(self, key: str, value) -> None:
        self.ctx.run("layer.set", layer=self.inspector.selected_id(), **{key: value})

    def _make(self, prop: P.Prop) -> QWidget:
        if prop.kind == "bool":
            return self.check(prop.title, lambda v, k=prop.key: self._set(k, v))
        if prop.kind == "colormap":
            box = _combo()
            box.setIconSize(QSize(48, 9))
            if prop.key == "colormap_negative":
                box.addItem("(none)", "")
            for name in colormaps.names():
                box.addItem(swatch_icon(name), name, name)
            box.setMaxVisibleItems(18)
            box.currentIndexChanged.connect(
                lambda _i, b=box, k=prop.key: self.inspector.run_guarded(
                    lambda v: self._set(k, v), b.currentData()))
            return box
        if prop.kind == "choice":
            return self.combo(prop.choices, lambda v, k=prop.key: self._set(k, v))
        if prop.kind == "window":
            rc = RangeControl()
            rc.range_changed.connect(
                lambda lo, hi, k=prop.key: self.inspector.run_guarded(
                    lambda w: self._set(k, w), (lo, hi)))
            rc.pressed.connect(self.ctx.store.begin_gesture)
            rc.released.connect(self.ctx.store.end_gesture)
            rc.reset_requested.connect(lambda k=prop.key: self._reset_window(k))
            if prop.key == "window":
                holder = QWidget()
                v = QVBoxLayout(holder)
                v.setContentsMargins(0, 0, 0, 0)
                v.setSpacing(4)
                v.addWidget(rc)
                row = QHBoxLayout()
                row.setSpacing(4)
                for text, tip, cmd, params in (
                    ("Auto", "The 1st to 99th percentile of this volume", "window.robust", {}),
                    ("Series", "A window for the whole series (4-D)", "window.robust",
                     {"series": True}),
                    ("Full", "The smallest to the largest value", "window.full", {}),
                ):
                    b = QPushButton(text)
                    b.setObjectName("tb-btn")
                    b.setToolTip(tip)
                    b.clicked.connect(lambda _c=False, c=cmd, pr=params: self.ctx.run(
                        c, layer=self.inspector.selected_id(), **pr))
                    row.addWidget(b)
                    if text == "Series":
                        self._series_button = b
                row.addStretch(1)
                v.addLayout(row)
                holder.range = rc
                return holder
            return rc
        log = prop.key == "gamma"
        return self.number(prop.lo, prop.hi, step=prop.step, unit=prop.unit,
                           default={"gamma": 1.0, "opacity": 1.0, "outline_px": 0.0}.get(prop.key),
                           log=log, on_change=lambda v, k=prop.key: self._set(k, v))

    def _reset_window(self, key: str) -> None:
        """Double-click on a window bar: the image's own robust range, or for
        the negative tail the positive window mirrored."""
        if key == "window":
            self.ctx.run("window.robust", layer=self.inspector.selected_id())
        else:
            self.ctx.run("layer.set", layer=self.inspector.selected_id(),
                         window_negative=self._mirror())

    def _mirror(self):
        layer = self.inspector.selected_layer()
        main = layer.display.window if layer is not None else None
        main = main or (0.0, 1.0)
        return (abs(float(main[0])), abs(float(main[1])))

    def applies(self) -> bool:
        layer = self.inspector.selected_layer()
        if layer is None:
            return False
        src = views.source_of(self.ctx.store, layer)
        ctx = P.layer_context(layer, src)
        ctx["overlay"] = self.inspector.is_overlay(layer.id)
        return any(A.evaluate(p.when, ctx) for p in self.props)

    def control(self, key: str) -> Optional[QWidget]:
        return self._controls.get(key)

    def sync(self) -> None:
        layer = self.inspector.selected_layer()
        if layer is None:
            return
        src = views.source_of(self.ctx.store, layer)
        ctx = P.layer_context(layer, src)
        ctx["overlay"] = self.inspector.is_overlay(layer.id)
        for prop in self.props:
            shown = A.evaluate(prop.when, ctx)
            self.show_row(prop.key, shown)
            if not shown:
                continue
            value = getattr(layer if prop.target == "layer" else layer.display, prop.key)
            widget = self._controls[prop.key]
            if prop.kind == "bool":
                _set_check(widget, value)
            elif prop.kind in ("colormap", "choice"):
                _set_combo(widget, value)
            elif prop.kind == "window":
                rc = widget.range if hasattr(widget, "range") else widget
                rc.set_window(*self._effective_window(prop.key, value, layer, src))
                if src is not None and not src.is_rgb:
                    t = views.frame_of(self.ctx.store, layer, src)
                    domain = src.data_range(t)
                    if domain is not None:
                        rc.set_data(domain, src.histogram(t))
                if prop.key == "window" and hasattr(self, "_series_button"):
                    self._series_button.setVisible(bool(src is not None and src.is_4d))
            else:
                widget.set_value(float(value))

    def _effective_window(self, key, value, layer, src):
        if value is not None:
            return value
        main = layer.display.window
        if main is None and src is not None:
            main = src.robust_range(views.frame_of(self.ctx.store, layer, src))
        main = main or (0.0, 1.0)
        if key == "window_negative":
            return (abs(float(main[0])), abs(float(main[1])))
        return main


class QualitySection(Section):
    """What a quality map says, for the selected one: its summary over the
    head, how it was made, how to read it, and its scale (relative to this
    run, or fixed so that runs can be compared)."""

    #: Fixed scales that make two runs' colours mean the same number.
    FIXED = {"tsnr": (0.0, 100.0)}

    def __init__(self, inspector) -> None:
        super().__init__(inspector, "quality", "QC map")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("sidecar-footer-summary")
        self.add_row("summary", None, self.summary, span=True)
        self.help = QLabel("")
        self.help.setWordWrap(True)
        self.help.setObjectName("sidecar-footer-summary")
        self.help.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.add_row("help", None, self.help, span=True)
        self.scale = self.combo((("run", "This run's range"),
                                 ("fixed", "Fixed (0 to 100)")), self._on_scale)
        self.add_row("scale", "Scale", self.scale,
                     "Relative: the colours use this run's own range. Fixed: the same "
                     "colour means the same value in every run.")

    def restore_defaults(self) -> None:
        self._on_scale("run")

    def _notes(self) -> dict:
        layer = self.inspector.selected_layer()
        src = views.source_of(self.ctx.store, layer) if layer is not None else None
        return getattr(src, "notes", {}) or {}

    def applies(self) -> bool:
        return "qc" in self._notes()

    def _on_scale(self, value) -> None:
        notes = self._notes()
        which = notes.get("qc")
        if value == "fixed" and which in self.FIXED:
            window = self.FIXED[which]
        else:
            # The window the map opened with: this run's own range.
            window = tuple(notes.get("window") or (0.0, 1.0))
        self.ctx.run("layer.set", layer=self.inspector.selected_id(), window=window)

    def sync(self) -> None:
        notes = self._notes()
        which = notes.get("qc", "")
        self.summary.setText(notes.get("summary", ""))
        self.help.setText(notes.get("help", ""))
        self.show_row("scale", which in self.FIXED)
        layer = self.inspector.selected_layer()
        if layer is not None and which in self.FIXED:
            fixed = tuple(layer.display.window or ()) == self.FIXED[which]
            _set_combo(self.scale, "fixed" if fixed else "run")


# ---------------------------------------------------------------------------
# View and layout
# ---------------------------------------------------------------------------


class ViewSection(Section):
    #: The display conventions, as the actions that toggle them.
    FLAGS = ("view.labels", "view.ras", "view.radiological", "view.world",
             "view.colorbar", "view.crosshair")

    def __init__(self, inspector) -> None:
        super().__init__(inspector, "view", "Slice views")
        self._boxes: dict[str, QCheckBox] = {}
        for action_id in self.FLAGS:
            d = A.ACTION_BY_ID[action_id]
            box = self.check(d.short or d.title, lambda _v, a=action_id: self._toggle(a))
            box.setToolTip(f"{d.title}. {d.help}" if d.help else d.title)
            self._boxes[action_id] = box
            self.add_row(action_id, None, box)
        from ..settings_pages import ColourButton

        self.cross_colour = ColourButton()
        self.cross_colour.setMinimumWidth(0)
        self.cross_colour.setFixedSize(96, 24)
        self.cross_colour.picked.connect(lambda c: self._cross(color=c))
        self.add_row("cross.colour", "Crosshair colour", self.cross_colour)
        self.cross_width = self.number(1, 5, step=1, unit="px", default=1,
                                       on_change=lambda v: self._cross(thickness=int(v)))
        self.add_row("cross.width", "Crosshair width", self.cross_width)
        self.cross_gap = self.number(0, 40, step=1, unit="px", default=0,
                                     on_change=lambda v: self._cross(gap=int(v)))
        self.add_row("cross.gap", "Gap at the centre", self.cross_gap,
                     "Pixels left empty around the centre, so the voxel under "
                     "the crosshair stays visible.")

    def restore_defaults(self) -> None:
        """Display conventions and the crosshair's look as installed."""
        from ....viz.settings import CrosshairSettings

        self.ctx.run("display.reset")
        self.ctx.settings_hub.update(lambda s: setattr(s, "crosshair", CrosshairSettings()))

    def _toggle(self, action_id: str) -> None:
        run = self.ctx.run_action
        if run is not None:
            run(action_id)

    def _cross(self, **values) -> None:
        def mutate(s):
            for k, v in values.items():
                setattr(s.crosshair, k, v)

        self.ctx.settings_hub.update(mutate)

    def sync(self) -> None:
        ctx = self.inspector.action_context()
        for action_id, box in self._boxes.items():
            d = A.ACTION_BY_ID[action_id]
            _set_check(box, A.evaluate(d.checked, ctx))
        cs = self.ctx.settings.crosshair
        self.cross_colour.set_value(cs.color)
        self.cross_width.set_value(cs.thickness)
        self.cross_gap.set_value(cs.gap)


class LayoutSection(Section):
    def __init__(self, inspector) -> None:
        super().__init__(inspector, "layout", "Layout")
        self.arrangement = self.combo(
            (("auto", "Automatic"), ("row", "In a row"), ("column", "In a column"),
             ("grid", "In a grid")),
            lambda v: self.ctx.run("layout.set", arrangement=v))
        self.add_row("arrangement", "Arrangement", self.arrangement,
                     "How the views of the multi-planar layouts are placed. Automatic "
                     "picks whichever of row, column or grid makes the images largest "
                     "for the viewer's shape.")
        planes = QWidget()
        row = QHBoxLayout(planes)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._plane_boxes: dict[str, QCheckBox] = {}
        for plane, text in (("sagittal", "Sag."), ("coronal", "Cor."), ("axial", "Ax.")):
            box = self.check(text, lambda _v: self._planes())
            box.setToolTip(f"Show the {plane} plane in the multi-planar layouts")
            self._plane_boxes[plane] = box
            row.addWidget(box)
        row.addStretch(1)
        self.add_row("planes", "Planes shown", planes,
                     "Which planes the multi-planar layouts show (at least one).")
        self.hero = self.combo(
            (("", "The chosen plane"), ("sagittal", "Sagittal"), ("coronal", "Coronal"),
             ("axial", "Axial"), ("render", "3-D")),
            lambda v: self.ctx.run("layout.set", hero=v))
        self.add_row("hero", "Large view", self.hero,
                     "In the large-view layout, which view is drawn large.")
        self.hero_size = self.number(30, 85, step=1, unit="%", default=62,
                                     on_change=lambda v: self.ctx.run(
                                         "layout.set", hero_fraction=v / 100.0))
        self.add_row("hero_size", "Large view size", self.hero_size,
                     "Its share of the viewer. You can also drag the gap beside it.")
        self.hero_side = self.combo((("left", "On the left"), ("top", "At the top")),
                                    lambda v: self.ctx.run("layout.set", hero_side=v))
        self.add_row("hero_side", "Large view position", self.hero_side)
        self.graph_side = self.combo((("bottom", "Below the views"),
                                      ("right", "Right of the views")),
                                     lambda v: self.ctx.run("layout.set", graph=v))
        self.add_row("graph", "Time course position", self.graph_side,
                     "Where the time-course graph of a 4-D image goes.")

    def restore_defaults(self) -> None:
        self.ctx.run("layout.reset")

    def _planes(self) -> None:
        planes = [p for p, box in self._plane_boxes.items() if box.isChecked()]
        if not planes:
            # Never none: put the box back.
            self.sync()
            return
        self.ctx.run("layout.set", planes=planes)

    def sync(self) -> None:
        lay = self.ctx.scene.layout
        _set_combo(self.arrangement, lay.arrangement)
        for plane, box in self._plane_boxes.items():
            _set_check(box, plane in lay.planes)
        _set_combo(self.hero, lay.hero)
        self.hero_size.set_value(round(lay.hero_fraction * 100))
        _set_combo(self.hero_side, lay.hero_side)
        _set_combo(self.graph_side, lay.graph)
        mode = self.ctx.scene.mode
        for key in ("hero", "hero_size", "hero_side"):
            self.show_row(key, mode == "hero")
        for key in ("arrangement", "planes"):
            self.show_row(key, mode in ("multi", "combo"))
        self.show_row("graph", self.inspector.has_series())

    def applies(self) -> bool:
        return self.ctx.scene.mode != "mosaic"


# ---------------------------------------------------------------------------
# 3-D
# ---------------------------------------------------------------------------


class MosaicSection(Section):
    """A mosaic as a figure is described: one plane, a grid of slices across
    the head, labels, a reference slice. The grammar line it builds is shown
    and can be written by hand for its extras (rows of different planes,
    renders)."""

    def __init__(self, inspector) -> None:
        super().__init__(inspector, "mosaic", "Mosaic", open_=False)
        run = self._set
        self.plane = self.combo((("axial", "Axial"), ("coronal", "Coronal"),
                                 ("sagittal", "Sagittal")), lambda v: run(plane=v))
        self.add_row("plane", "Plane", self.plane, "The plane every tile of the grid cuts.")
        self.rows = self.number(1, 10, step=1, default=3, on_change=lambda v: run(rows=int(v)))
        self.add_row("rows", "Rows", self.rows)
        self.cols = self.number(1, 16, step=1, default=6, on_change=lambda v: run(cols=int(v)))
        self.add_row("cols", "Columns", self.cols)
        self.start = self.number(-200, 200, step=0.5, unit="mm",
                                 on_change=lambda v: run(start=float(v)))
        self.add_row("start", "From", self.start,
                     "The first slice's side of the range, in mm along the plane's axis "
                     "(scanner coordinates). The slices are spread evenly across it.")
        self.end = self.number(-200, 200, step=0.5, unit="mm",
                               on_change=lambda v: run(end=float(v)))
        self.add_row("end", "To", self.end)
        fit = QPushButton("Fit to the head")
        fit.setObjectName("tb-btn")
        fit.setToolTip("Spread the slices across the head found in the image (for an axial "
                       "grid of a head with its neck, the top 140 mm: the brain).")
        fit.clicked.connect(lambda: self.ctx.run("mosaic.set", fit=True))
        self.add_row("fit", None, fit, span=True)
        self.crop = self.check("Crop the tiles to the head", lambda v: run(crop=v))
        self.add_row("crop", None, self.crop,
                     "Leave out the empty field of view around the head, the same crop for "
                     "every tile so they stay alike.")
        self.labels = self.check("Slice positions under the tiles", lambda v: run(labels=v))
        self.add_row("labels", None, self.labels)
        self.reference = self.check("Reference slice", lambda v: run(reference=v))
        self.add_row("reference", None, self.reference,
                     "A perpendicular slice before the grid with a line where every tile "
                     "cuts, so a reader sees where the slices are.")
        self.overlap = self.number(0.0, 0.5, step=0.05, default=0.0,
                                   on_change=lambda v: run(overlap=float(v)))
        self.add_row("overlap", "Overlap", self.overlap,
                     "Neighbouring tiles overlap by this fraction of their width: a "
                     "compact figure.")
        self.line = QLineEdit()
        self.line.setToolTip(
            "The mosaic as a line (NiiVue's grammar), to write by hand for what the "
            "builder does not do: A, C, S choose the plane; numbers are positions in mm; "
            "';' starts a row; X draws where other tiles cut; L- hides the labels; H 0.3 "
            "overlaps tiles. Written by hand, the builder leaves it alone until a "
            "control above is used.")
        self.line.editingFinished.connect(
            lambda: self.ctx.run("view.mosaic_text", text=self.line.text()))
        self.add_row("line", "Line", self.line)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 2, 0, 0)
        row.setSpacing(4)
        self.show_button = QPushButton("Show the mosaic")
        self.show_button.setObjectName("tb-btn")
        self.show_button.clicked.connect(lambda: self.ctx.run("view.mode", mode="mosaic"))
        row.addWidget(self.show_button)
        save = QPushButton("Save the figure...")
        save.setObjectName("tb-btn-primary")
        save.setToolTip("The mosaic as a PNG, at two to four times the screen's resolution "
                        "for print, on black or transparent")
        save.clicked.connect(lambda: self.inspector.presenter.save_mosaic_figure())
        row.addWidget(save)
        row.addStretch(1)
        self.add_row("buttons", None, buttons, span=True)

    def _set(self, **params) -> None:
        try:
            self.ctx.run("mosaic.set", **params)
        except ValueError as exc:
            self.ctx.status.emit(str(exc))

    def restore_defaults(self) -> None:
        from ....viz.scene import MosaicBuild

        d = MosaicBuild()
        self.ctx.run("mosaic.set", plane=d.plane, rows=d.rows, cols=d.cols, labels=d.labels,
                     reference=d.reference, overlap=d.overlap, crop=d.crop, fit=True)

    def sync(self) -> None:
        from ....viz.commands.volume import mosaic_extent

        b = self.ctx.scene.mosaic_build
        _set_combo(self.plane, b.plane)
        self.rows.set_value(b.rows)
        self.cols.set_value(b.cols)
        _set_check(self.labels, b.labels)
        _set_check(self.crop, b.crop)
        _set_check(self.reference, b.reference)
        self.overlap.set_value(b.overlap)
        fit = mosaic_extent(self.ctx.store, b.plane)
        if fit is not None:
            pad = max(20.0, 0.3 * (fit[1] - fit[0]))
            lo, hi = fit[0] - pad, fit[1] + pad
            for ctl in (self.start, self.end):
                ctl.set_range(round(lo, 1), round(hi, 1))
        self.start.set_value(b.start if b.start is not None else (fit[0] if fit else 0.0))
        self.end.set_value(b.end if b.end is not None else (fit[1] if fit else 0.0))
        if self.line.text() != self.ctx.scene.mosaic and not self.line.hasFocus():
            self.line.setText(self.ctx.scene.mosaic)
        self.show_button.setVisible(self.ctx.scene.mode != "mosaic")


class RenderSection(Section):
    """The 3-D parameters of one purpose group."""

    def __init__(self, inspector, group: str) -> None:
        super().__init__(inspector, f"3d.{group}", render3d.PARAM_GROUPS[group],
                         open_=group in ("look", "transfer", "overlays"))
        self.group = group
        self.params = [p for p in render3d.PARAMS if p.group == group]
        self._controls: dict[str, QWidget] = {}
        if group == "look":
            self.effect = self.combo([(e, e) for e in render3d.EFFECTS],
                                     lambda v: self.ctx.run("render.effect", effect=v))
            self.add_row("effect", "Rendering effect", self.effect,
                         "How the volume is turned into a picture: shaded surfaces "
                         "(Standard, Matte, Realistic, Topography), translucent shells "
                         "(Glass, Edges, Shell), a maximum intensity projection (MIP), an "
                         "X-ray, or opacity peeling. Each keeps its own parameters.")
            self.use_colormap = self.check(
                "Use the 2-D colour map",
                lambda v: self.ctx.run("render.use_colormap", value=v))
            self.add_row("use_colormap", None, self.use_colormap,
                         "Colour the render with the image's colour map, window and gamma, "
                         "as the slices are coloured; the cut surface is then the slice "
                         "itself. Off: the effect's own colouring.")
        for param in self.params:
            self._controls[param.key] = self._make(param)
            self.add_row(param.key, None if param.kind == "check" else param.label,
                         self._controls[param.key], param.help)
        if group == "look":
            buttons = QWidget()
            row = QHBoxLayout(buttons)
            row.setContentsMargins(0, 4, 0, 0)
            row.setSpacing(4)
            for text, cmd, tip in (
                ("Reset this effect", "render.reset_params",
                 "This effect's parameters back to its defaults or preset."),
                ("Reset all", "render.reset_all", "Every effect's parameters."),
                ("Reset view", "render.reset_view", "Put the camera back."),
            ):
                b = QPushButton(text)
                b.setObjectName("tb-btn")
                b.setToolTip(tip)
                b.clicked.connect(lambda _c=False, c=cmd: self.ctx.run(c))
                row.addWidget(b)
            row.addStretch(1)
            self.add_row("reset", None, buttons, span=True)

    def restore_defaults(self) -> None:
        """This group's parameters back to the effect's own (the appearance
        group also puts the effect and the colour source back)."""
        from ....viz.scene import RenderState

        with self.ctx.store.gesture():
            if self.group == "look":
                fresh = RenderState()
                self.ctx.run("render.effect", effect=fresh.effect)
                self.ctx.run("render.use_colormap", value=fresh.use_colormap)
            self.ctx.run("render.reset_params", keys=[p.key for p in self.params])

    def _make(self, param: render3d.Param) -> QWidget:
        key = param.key
        if param.kind == "choice":
            return self.combo([(i, name) for i, name in enumerate(render3d.LIGHTINGS)],
                              lambda v, k=key: self.ctx.run("render.param", key=k,
                                                            value=float(v)))
        if param.kind == "check":
            return self.check(param.label, lambda v, k=key: self.ctx.run(
                "render.param", key=k, value=1.0 if v else 0.0))
        scale = param.scale or 1.0
        step = 1.0 / scale if scale > 1 else 1.0
        return self.number(param.lo / scale, param.hi / scale, step=step,
                           default=param.default / scale,
                           on_change=lambda v, k=key, sc=scale: self.ctx.run(
                               "render.param", key=k, value=float(v) * sc))

    def control(self, key: str) -> Optional[QWidget]:
        return self._controls.get(key) or (self.effect if key == "effect" and
                                           self.group == "look" else None)

    def used(self) -> set[str]:
        return render3d.EFFECT_PARAMS.get(self.ctx.scene.render.effect, set())

    def applies(self) -> bool:
        if not self.inspector.render_on_screen():
            return False
        return self.group == "look" or any(p.key in self.used() for p in self.params)

    def sync(self) -> None:
        rs = self.ctx.scene.render
        values = render3d.values_for(rs)
        used = self.used()
        if self.group == "look":
            _set_combo(self.effect, rs.effect)
            _set_check(self.use_colormap, rs.use_colormap)
        for param in self.params:
            shown = param.key in used
            self.show_row(param.key, shown)
            if not shown:
                continue
            w = self._controls[param.key]
            v = values.get(param.key, param.default)
            if param.kind == "choice":
                _set_combo(w, int(v))
            elif param.kind == "check":
                _set_check(w, v >= 0.5)
            else:
                w.set_value(float(v) / (param.scale or 1.0))


class ClipSection(Section):
    def __init__(self, inspector) -> None:
        super().__init__(inspector, "3d.clip", "3-D: clip planes", open_=False)
        self.preset = self.combo(CLIP_PRESETS, lambda v: self.ctx.run("clip.preset", preset=v))
        self.add_row("preset", "Preset", self.preset,
                     "Replace the clip planes with an arrangement. At the crosshair, "
                     "the cut faces are the slices the 2-D views show.")
        self.at_cursor = self.check("Through the crosshair",
                                    lambda v: self.ctx.run("clip.mode", at_cursor=v))
        self.at_cursor.setToolTip("Every plane passes through the crosshair, keeping its "
                                  "angle, and follows it.")
        self.add_row("at_cursor", None, self.at_cursor)
        self.cut_away = self.check("Remove only where all planes cut",
                                   lambda v: self.ctx.run("clip.mode", cut_away=v))
        self.cut_away.setToolTip("On: remove only what every plane cuts (a wedge, a "
                                 "corner). Off: remove what any plane cuts (a crop).")
        self.add_row("cut_away", None, self.cut_away)
        self.choice = self.combo([(i, f"Plane {i + 1}") for i in range(render3d.MAX_CLIP_PLANES)],
                                 self._choose)
        self.add_row("choice", "Plane to adjust", self.choice,
                     "Which plane the controls below, the Shift keys and the "
                     "gestures act on.")
        self.enabled = self.check("Plane active", lambda v: self._clip(active=v))
        self.add_row("enabled", None, self.enabled)
        self.az = self.number(0, 360, step=1, unit="deg", default=0,
                              on_change=lambda v: self._clip(az=v))
        self.add_row("az", "Azimuth", self.az,
                     "The plane's direction around the vertical axis (degrees).")
        self.el = self.number(-90, 90, step=1, unit="deg", default=0,
                              on_change=lambda v: self._clip(el=v))
        self.add_row("el", "Elevation", self.el,
                     "The plane's tilt above or below the horizontal (degrees).")
        self.pos = self.number(0, 100, step=1, unit="%", default=50,
                               on_change=lambda v: self._clip(pos=v / 100.0))
        self.add_row("pos", "Depth", self.pos,
                     "Where the plane cuts, from one side of the volume (0 %) to the "
                     "other (100 %).")
        self.thick = self.number(0, 100, step=1, unit="%", default=100,
                                 on_change=lambda v: self._clip(thick=v / 100.0))
        self.add_row("thick", "Slab thickness", self.thick,
                     "Below 100 %, only a slab this thick is removed (a slot), not "
                     "everything beyond the plane.")

    def restore_defaults(self) -> None:
        with self.ctx.store.gesture():
            self.ctx.run("clip.preset", preset="none")
            self.ctx.run("clip.mode", cut_away=False, at_cursor=False)

    def _choose(self, index) -> None:
        index = int(index)
        if index != self.ctx.active_clip:
            self.ctx.active_clip = index
            self.ctx.active_clip_changed.emit(index)
            self.sync()

    def _clip(self, **params) -> None:
        self.ctx.run("clip.set", index=self.ctx.active_clip, **params)

    def applies(self) -> bool:
        return self.inspector.render_on_screen()

    def sync(self) -> None:
        clips = self.ctx.scene.clips
        rs = self.ctx.scene.render
        ci = self.ctx.active_clip
        _set_combo(self.choice, ci)
        clip = clips[ci] if ci < len(clips) else ClipPlane()
        _set_check(self.enabled, clip.active)
        _set_check(self.at_cursor, rs.cut_at_cursor)
        _set_check(self.cut_away, rs.cut_away)
        if not any(c.active for c in clips):
            _set_combo(self.preset, "none")
        for w, v in ((self.az, clip.az), (self.el, clip.el), (self.pos, clip.pos * 100.0),
                     (self.thick, clip.thick * 100.0)):
            w.set_value(v)
            w.setEnabled(clip.active)
        self.pos.setEnabled(clip.active and not rs.cut_at_cursor)


# ---------------------------------------------------------------------------
# The column
# ---------------------------------------------------------------------------


class Inspector(QWidget):
    """The controls column of the volume viewer."""

    #: "Add..." was pressed: the host chooses the file (and reads it).
    add_requested = pyqtSignal()

    def __init__(self, ctx: ViewerContext, presenter, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.presenter = presenter
        self.setObjectName("viz-inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.syncing = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 8)
        v.setSpacing(0)
        self.sections: dict[str, Section] = {}
        # By purpose: the layers and how the selected one is drawn; the
        # slice views and their arrangement; then the 3-D, the clip planes
        # beside the cut surface they make.
        render_groups = list(render3d.PARAM_GROUPS)
        builders = [
            LayersSection,
            lambda i: LayerPropsSection(i, "display", "Layer display", "display"),
            lambda i: LayerPropsSection(i, "overlay", "Overlay options", "overlay"),
            QualitySection,
            ViewSection,
            LayoutSection,
            MosaicSection,
        ]
        for g in render_groups:
            builders.append(lambda i, g=g: RenderSection(i, g))
            if g == "cut":
                builders.append(ClipSection)
        for build in builders:
            section = build(self)
            self.sections[section.key] = section
            v.addWidget(section)
        v.addStretch(1)
        ctx.qstore.changed.connect(self._on_changed)
        ctx.layer_selected.connect(lambda _l: self.sync())
        from ..bridge import connect_while_alive

        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w._on_settings())
        self.sync()

    # -- what the sections ask -------------------------------------------
    def selected_layer(self):
        scene = self.ctx.scene
        layer = scene.layer(self.ctx.selected_layer) if self.ctx.selected_layer else None
        return layer if layer is not None else scene.base_layer()

    def selected_id(self) -> str:
        layer = self.selected_layer()
        return layer.id if layer is not None else ""

    def is_overlay(self, layer_id: str) -> bool:
        base = self.ctx.scene.base_layer()
        return base is not None and layer_id != base.id

    def render_on_screen(self) -> bool:
        return bool(self.presenter.canvases("render")) or (
            self.presenter.render_allowed() and self.ctx.scene.mode in ("3d", "combo", "hero"))

    def has_series(self) -> bool:
        src = self.presenter.source
        return bool(src is not None and src.is_4d)

    def action_context(self) -> dict:
        return self.presenter.action_context()

    def section(self, key: str) -> Section:
        return self.sections[key]

    def run_guarded(self, fn: Callable[[Any], None], value) -> None:
        """Run a control's change unless the column is being synced from the
        scene (a sync must never run a command)."""
        if self.syncing:
            return
        try:
            fn(value)
        except ValueError as exc:
            self.ctx.status.emit(str(exc))

    # -- scene -> controls ------------------------------------------------
    def _on_changed(self, paths) -> None:
        if not self.isVisible():
            return
        self.sync()

    def _on_settings(self) -> None:
        # Rare, so always: a column opened later must not show stale values.
        self.sync()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.sync()

    def sync(self) -> None:
        if self.ctx.selected_layer and self.ctx.scene.layer(self.ctx.selected_layer) is None:
            self.ctx.selected_layer = ""
        self.syncing = True
        try:
            for section in self.sections.values():
                on = section.applies()
                if section.isHidden() == on:
                    section.setVisible(on)
                if on:
                    section.sync()
        finally:
            self.syncing = False


__all__ = ["CLIP_PRESETS", "Inspector", "Section", "swatch_icon"]

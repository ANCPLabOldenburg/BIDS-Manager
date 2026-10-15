"""The viewer's Settings pages: Viewer, Quality control, and Viewer shortcuts.

**Viewer** and **Quality control** are GENERATED from
:mod:`bidsmgr.viz.settings` (``VIEWER_SECTIONS``, ``QC_SECTIONS``; the MRI
quality checks' own model is :mod:`bidsmgr.qc.config`): every field becomes
a control chosen by its type (a check box, a choice, a number with its
range and unit, a colour, a colour map), titled and explained by the
field's own metadata. A preference added to the model appears here with no
widget code. Each group has its own Restore defaults, and a field whose
hint names ``enabled_by`` is enabled only while that switch is on.

**Viewer shortcuts** edits the keyboard map (record a key, add a second one,
unbind, reset one or all, import and export as JSON, conflicts shown as they
arise) and the mouse map (which tool each button, modifier and wheel runs on
a slice and on the 3-D view).

Both pages work on a COPY of the settings and hand back the edited copy;
the dialog decides when to adopt it (Save), so Cancel costs nothing.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QKeySequenceEdit,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...viz import actions as A
from ...viz import inputmap, keynames
from ...viz.compute import colormaps
from ...viz.settings import QC_SECTIONS, VIEWER_SECTIONS, VizSettings
from ..widgets.splitter import Splitter


# ---------------------------------------------------------------------------
# Generated controls
# ---------------------------------------------------------------------------


class ColourButton(QPushButton):
    """A swatch that opens a colour dialog. ``allow_empty`` adds a way back
    to "no colour" (for preferences where empty means "by type")."""

    #: The user picked a colour (never emitted by :meth:`set_value`).
    picked = pyqtSignal(str)

    def __init__(self, empty_text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._value = ""
        self._empty_text = empty_text
        self.setObjectName("tb-btn")
        self.setMinimumWidth(110)
        self.clicked.connect(self._pick)

    def value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value or ""
        if self._value:
            col = QColor(self._value)
            text = "#000000" if col.lightness() > 140 else "#ffffff"
            self.setText(col.name())
            self.setStyleSheet(
                f"QPushButton {{ background: {col.name()}; color: {text};"
                f" border-radius: 4px; padding: 3px 10px; }}"
            )
        else:
            self.setText(self._empty_text or "Choose...")
            self.setStyleSheet("")

    def _pick(self) -> None:
        start = QColor(self._value) if self._value else QColor("#4FC3F7")
        dlg = QColorDialog(start, self)
        if dlg.exec() == QColorDialog.DialogCode.Accepted and dlg.currentColor().isValid():
            self.set_value(dlg.currentColor().name())
            self.picked.emit(self._value)


class _Control:
    """One generated control: its widget and how to read and write it."""

    def __init__(self, widget: QWidget, get: Callable[[], Any], put: Callable[[Any], None]):
        self.widget = widget
        self.get = get
        self.put = put


def _control_for(field_info) -> Optional[_Control]:
    annotation = field_info.annotation
    extra = field_info.json_schema_extra or {}
    kind = extra.get("kind")
    origin = typing.get_origin(annotation)
    if origin is dict or annotation is dict:
        return None                       # specialist editors handle these
    if annotation is bool:
        w = QCheckBox()
        return _Control(w, w.isChecked, lambda v, w=w: w.setChecked(bool(v)))
    if origin is typing.Literal:
        w = QComboBox()
        labels = extra.get("labels", {})
        for value in typing.get_args(annotation):
            w.addItem(labels.get(value, str(value).capitalize()), value)

        def put(v, w=w):
            i = w.findData(v)
            w.setCurrentIndex(max(i, 0))

        return _Control(w, lambda w=w: w.currentData(), put)
    if kind == "colour":
        button = ColourButton(extra.get("empty", ""))
        if not extra.get("empty"):
            return _Control(button, button.value, button.set_value)
        # A colour that may be empty ("By type") needs a way back to empty.
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(button)
        clear = QPushButton("Clear")
        clear.setObjectName("tb-btn")
        clear.clicked.connect(lambda: button.set_value(""))
        row.addWidget(clear)
        row.addStretch(1)
        return _Control(holder, button.value, button.set_value)
    if kind == "colormap":
        w = QComboBox()
        w.addItems(colormaps.names())
        return _Control(w, w.currentText, lambda v, w=w: w.setCurrentText(str(v)))
    if annotation in (int, float):
        lo, hi = extra.get("range", (0, 10 ** 6))
        if annotation is int:
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(extra.get("step", 1)))
        else:
            w = QDoubleSpinBox()
            step = float(extra.get("step", 0.1))
            w.setDecimals(max(0, len(f"{step:g}".split(".")[1]) if "." in f"{step:g}" else 0))
            w.setRange(float(lo), float(hi))
            w.setSingleStep(step)
        if extra.get("unit"):
            w.setSuffix(f" {extra['unit']}")
        if extra.get("zero"):
            w.setSpecialValueText(str(extra["zero"]))
        w.setMinimumWidth(110)
        return _Control(w, w.value, lambda v, w=w: w.setValue(v))
    if annotation is str:
        w = QLineEdit()
        return _Control(w, w.text, lambda v, w=w: w.setText(str(v)))
    return None


class GeneratedSettingsPage(QWidget):
    """Every field of ``sections`` (of :class:`VizSettings`), one group
    each, generated from the model. Rows wrap (the label above its field)
    when the page is narrow, so nothing here sets a width the dialog cannot
    shrink below."""

    def __init__(self, sections: tuple[str, ...], *, hint: str = "", parent=None,
                 root=None) -> None:
        super().__init__(parent)
        self._sections = tuple(sections)
        # The model the sections belong to (the viewer settings unless another
        # pydantic settings model is given, such as the classifier's).
        self._root = root if root is not None else VizSettings
        self._controls: dict[tuple[str, str], _Control] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("settings-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(10)
        self.body_layout = lay
        if hint:
            label = QLabel(hint)
            label.setObjectName("dlg-hint")
            label.setWordWrap(True)
            lay.addWidget(label)
        for section in self._sections:
            lay.addWidget(self._build_section(section))
        self.extend(lay)
        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def extend(self, lay: QVBoxLayout) -> None:
        """Add a page's own groups after the generated ones."""

    def _build_section(self, section: str) -> QWidget:
        model_cls = self._root.model_fields[section].annotation
        box = QGroupBox(model_cls.model_config.get("title", section.capitalize()))
        form = QFormLayout(box)
        # Fields keep their natural size (a spin box stretched across the
        # page puts its number and its arrows a window apart); a row too wide
        # for the page puts its field under its label.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        switches: dict[str, QCheckBox] = {}
        for name, info in model_cls.model_fields.items():
            control = _control_for(info)
            if control is None:
                continue
            label = QLabel(info.title or name)
            if info.description:
                label.setToolTip(info.description)
                control.widget.setToolTip(info.description)
            control.widget.setProperty("viz_setting", f"{section}.{name}")
            form.addRow(label, control.widget)
            self._controls[(section, name)] = control
            if isinstance(control.widget, QCheckBox):
                switches[name] = control.widget
            parent = (info.json_schema_extra or {}).get("enabled_by")
            if parent in switches:
                switch = switches[parent]
                for w in (label, control.widget):
                    switch.toggled.connect(w.setEnabled)
        # This group alone back to its defaults (not saved until Save).
        reset = QPushButton("Restore defaults")
        reset.setObjectName("tb-btn")
        reset.setToolTip(f"Set the {box.title().lower()} fields back to their defaults "
                         "(saved with Save)")
        reset.clicked.connect(lambda _c=False, sec=section: self.restore_section(sec))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(reset)
        form.addRow(row)
        return box

    def restore_section(self, section: str) -> None:
        """``section``'s controls back to the model's defaults."""
        defaults = self._root.model_fields[section].annotation()
        for (sec, name), control in self._controls.items():
            if sec == section:
                control.put(getattr(defaults, name))

    def load(self, settings) -> None:
        for (section, name), control in self._controls.items():
            control.put(getattr(getattr(settings, section), name))

    def apply_to(self, settings) -> None:
        """Write the controls into ``settings`` (a copy the dialog owns)."""
        for (section, name), control in self._controls.items():
            setattr(getattr(settings, section), name, control.get())

    def control(self, path: str) -> Optional[QWidget]:
        """The widget for ``section.field`` (tests)."""
        section, name = path.split(".", 1)
        found = self._controls.get((section, name))
        return found.widget if found else None


class ViewerSettingsPage(GeneratedSettingsPage):
    """Every viewer preference, generated from the settings model."""

    def __init__(self, parent=None) -> None:
        super().__init__(
            VIEWER_SECTIONS,
            hint="Every viewer follows these: the Editor's, comparisons, defacing "
                 "previews. Saved changes reach open viewers at once.", parent=parent)

    def extend(self, lay: QVBoxLayout) -> None:
        lay.addWidget(self._build_memory_box())

    def _build_memory_box(self) -> QWidget:
        """Each kind of image remembers its own layout, and panels their
        sizes; this is the one place to make them forget."""
        box = QGroupBox("Remembered views")
        v = QVBoxLayout(box)
        self.memory_text = QLabel("")
        self.memory_text.setObjectName("dlg-hint")
        self.memory_text.setWordWrap(True)
        v.addWidget(self.memory_text)
        row = QHBoxLayout()
        self.forget_button = QPushButton("Forget every remembered view")
        self.forget_button.setToolTip(
            "Layouts per kind of image, panel sizes, the 3-D look, trace and spectrum "
            "options: all back to their presets when you Save. Shortcuts and saved "
            "presets are kept.")
        self.forget_button.setObjectName("tb-btn")
        self.forget_button.clicked.connect(self._forget)
        row.addWidget(self.forget_button)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _forget(self) -> None:
        self.forget_layouts = True
        self._show_memory({})

    def _show_memory(self, counts: Optional[dict] = None) -> None:
        counts = counts or {}
        parts = []
        if counts.get("kinds"):
            parts.append(f"the layout of {counts['kinds']} kind(s) of image")
        if counts.get("look"):
            parts.append("the display and 3-D look")
        if counts.get("signals"):
            parts.append(f"trace options for {counts['signals']} kind(s) of signal")
        if counts.get("spectrum"):
            parts.append("the spectrum options")
        if counts.get("sizes"):
            parts.append(f"{counts['sizes']} panel size(s)")
        if parts:
            self.memory_text.setText(
                "Every view keeps how you left it, in any window: "
                + ", ".join(parts) + ". A kind you never arranged opens in its preset "
                "(a BOLD run with its time course, a T1 without).")
        else:
            self.memory_text.setText(
                "Nothing remembered: every kind of image opens in its preset "
                "(a BOLD run with its time course, a T1 without).")
        self.forget_button.setEnabled(bool(parts))

    #: Set by "Forget": the dialog then saves no layout memory.
    forget_layouts = False

    def load(self, settings: VizSettings) -> None:
        super().load(settings)
        from ...viz import memory

        self.forget_layouts = False
        self._show_memory(memory.remembered_count(settings))


class QualitySettingsPage(GeneratedSettingsPage):
    """How the quality checks run: when, with which methods, and every
    threshold of the MRI checks (BOLD, anatomical, diffusion) and of the
    MEG and EEG check."""

    def __init__(self, parent=None) -> None:
        super().__init__(
            QC_SECTIONS,
            hint="The viewer, the Editor's Quality check and the post-convert step all use "
                 "these; bidsmgr-qc reads the same settings from a JSON file "
                 "(--print-config writes one). Each result records the settings it was "
                 "made with, and a changed setting recomputes what is on screen. Hover a "
                 "setting for what it does.", parent=parent)

    def extend(self, lay: QVBoxLayout) -> None:
        box = QGroupBox("Tools on this computer")
        v = QVBoxLayout(box)
        self.tools_text = QLabel(self._tools_text())
        self.tools_text.setObjectName("dlg-hint")
        self.tools_text.setWordWrap(True)
        v.addWidget(self.tools_text)
        # After the methods, which it qualifies.
        lay.insertWidget(min(3, lay.count()), box)

    @staticmethod
    def _tools_text() -> str:
        from ...qc import tools

        why = tools.unavailable_reason()
        if why:
            return ("The networks and niimath cannot run here (" + why + "): every check "
                    "uses the fast methods, whatever is chosen above, and says so in its "
                    "result.")
        return ("mindgrab, robust_tissue (brainchop) and niimath can run here. A network "
                "is downloaded the first time it is used.")


# ---------------------------------------------------------------------------
# Keyboard and mouse
# ---------------------------------------------------------------------------


def _key_text(seq: QKeySequence) -> str:
    """Our spelling of the recorded key, for the keymap. Qt's portable text
    ("Shift+A", "Ctrl+Shift+Z", "PgUp") is the spelling the keymap stores on
    every OS; ``Ctrl`` is the Command key on macOS, and ``keynames.key``
    writes it as one there."""
    if seq.isEmpty():
        return ""
    return A.normalise_key(seq.toString(QKeySequence.SequenceFormat.PortableText))


class ShortcutsPage(QWidget):
    """The keyboard map and the mouse map."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keymap: dict[str, list[str]] = {}
        self._mousemap: dict[str, str] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        split = Splitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_keys())
        split.addWidget(self._build_mouse())
        split.setSizes([420, 260])
        outer.addWidget(split)

    # -- keyboard ----------------------------------------------------------

    def _build_keys(self) -> QWidget:
        box = QGroupBox("Keyboard")
        v = QVBoxLayout(box)
        hint = QLabel("Keys work while the viewer has focus (click or scroll an "
                      "image once). Select an action, press the key in the box, "
                      "then Set or Add.")
        hint.setObjectName("dlg-hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find an action or a key")
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Action", "Group", "Keys"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for d in A.ACTIONS:
            row = self.table.rowCount()
            self.table.insertRow(row)
            title = QTableWidgetItem(d.title)
            title.setData(Qt.ItemDataRole.UserRole, d.id)
            if d.help:
                title.setToolTip(d.help)
            self.table.setItem(row, 0, title)
            self.table.setItem(row, 1, QTableWidgetItem(d.category))
            self.table.setItem(row, 2, QTableWidgetItem(""))
        v.addWidget(self.table, 1)

        edit_row = QHBoxLayout()
        self.recorder = QKeySequenceEdit()
        self.recorder.setMaximumSequenceLength(1)
        self.recorder.setToolTip("Press the key to bind")
        edit_row.addWidget(self.recorder, 1)
        for text, slot, tip in (
            ("Set", self._set_key, "Replace this action's keys with the one recorded."),
            ("Add", self._add_key, "Give this action the recorded key as well."),
            ("Unbind", self._unbind, "This action gets no key."),
            ("Default", self._default_one, "This action's own keys."),
        ):
            b = QPushButton(text)
            b.setObjectName("tb-btn")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            edit_row.addWidget(b)
        v.addLayout(edit_row)

        self.conflicts = QLabel("")
        # The error colour comes from the stylesheet (palette token), keyed
        # on a property, so it follows a theme swap.
        self.conflicts.setObjectName("viz-conflicts")
        self.conflicts.setWordWrap(True)
        v.addWidget(self.conflicts)

        file_row = QHBoxLayout()
        for text, slot in (("Import...", self._import), ("Export...", self._export),
                           ("Reset all keys", self._reset_keys)):
            b = QPushButton(text)
            b.setObjectName("tb-btn")
            b.clicked.connect(slot)
            file_row.addWidget(b)
        file_row.addStretch(1)
        v.addLayout(file_row)
        return box

    def _selected_action(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def _recorded(self) -> str:
        return _key_text(self.recorder.keySequence())

    def _keys_of(self, action_id: str) -> list[str]:
        return A.effective_keymap(self._keymap).get(action_id, [])

    def _store(self, action_id: str, keys: list[str]) -> None:
        default = [A.normalise_key(k) for k in A.ACTION_BY_ID[action_id].keys]
        keys = [A.normalise_key(k) for k in keys if A.normalise_key(k)]
        if keys == default:
            self._keymap.pop(action_id, None)
        else:
            self._keymap[action_id] = keys
        self._refresh_keys()

    def _set_key(self) -> None:
        action_id, key = self._selected_action(), self._recorded()
        if action_id and key:
            self._store(action_id, [key])

    def _add_key(self) -> None:
        action_id, key = self._selected_action(), self._recorded()
        if action_id and key and key not in self._keys_of(action_id):
            self._store(action_id, self._keys_of(action_id) + [key])

    def _unbind(self) -> None:
        action_id = self._selected_action()
        if action_id:
            self._store(action_id, [])

    def _default_one(self) -> None:
        action_id = self._selected_action()
        if action_id:
            self._keymap.pop(action_id, None)
            self._refresh_keys()

    def _reset_keys(self) -> None:
        self._keymap = {}
        self._refresh_keys()

    def _refresh_keys(self) -> None:
        keymap = A.effective_keymap(self._keymap)
        clash = A.conflicts(keymap)
        clashing = {a for _k, a, b in clash} | {b for _k, a, b in clash}
        error = QColor(*_palette_token("error", "#f85149"))
        for row in range(self.table.rowCount()):
            action_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            item = self.table.item(row, 2)
            item.setText(", ".join(keynames.keys(keymap.get(action_id, []))))
            changed = action_id in self._keymap
            font = item.font()
            font.setBold(changed)
            item.setFont(font)
            item.setForeground(error if action_id in clashing else self.table.palette().text())
        if clash:
            lines = [f"{keynames.key(key)}: {A.ACTION_BY_ID[a].title} and "
                     f"{A.ACTION_BY_ID[b].title}"
                     for key, a, b in clash]
            self.conflicts.setText("Used twice where both apply (neither will work "
                                   "until one is changed):\n" + "\n".join(lines))
        else:
            self.conflicts.setText("No key is used twice.")
        state = "error" if clash else "ok"
        if self.conflicts.property("state") != state:
            self.conflicts.setProperty("state", state)
            self.conflicts.style().unpolish(self.conflicts)
            self.conflicts.style().polish(self.conflicts)
        self._filter(self.search.text())

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.table.rowCount()):
            hay = " ".join(keynames.search_words(self.table.item(row, c).text())
                           for c in range(3))
            self.table.setRowHidden(row, bool(needle) and needle not in hay)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export viewer shortcuts",
                                              "viewer_shortcuts.json", "JSON (*.json)")
        if path:
            self.export_to(Path(path))

    def export_to(self, path: Path) -> None:
        payload = {"keymap": A.effective_keymap(self._keymap), "mousemap": self._mousemap}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import viewer shortcuts", "",
                                              "JSON (*.json)")
        if path:
            try:
                self.import_from(Path(path))
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Could not import", str(exc))

    def import_from(self, path: Path) -> None:
        """Adopt a file written by Export (or by hand). Unknown actions are
        ignored; a key the file does not name keeps its current binding."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a shortcuts file")
        for action_id, keys in (data.get("keymap") or {}).items():
            if action_id in A.ACTION_BY_ID and isinstance(keys, list):
                self._store(action_id, [str(k) for k in keys])
        for key, tool in (data.get("mousemap") or {}).items():
            canvas, _, gesture = str(key).partition(":")
            if canvas in inputmap.GESTURES and tool in inputmap.tools_for(canvas, gesture):
                self._set_mouse(canvas, gesture, tool)
        self._refresh_keys()
        self._refresh_mouse()

    # -- mouse -------------------------------------------------------------

    def _build_mouse(self) -> QWidget:
        box = QGroupBox("Mouse")
        v = QVBoxLayout(box)
        self.mouse_table = QTableWidget(0, 3, self)
        self.mouse_table.setHorizontalHeaderLabels(["On", "Gesture", "Does"])
        self.mouse_table.verticalHeader().setVisible(False)
        self.mouse_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.mouse_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self._mouse_combos: dict[str, QComboBox] = {}
        for canvas, where in (("slice", "A slice"), ("render", "The 3-D view"),
                              ("traces", "Traces")):
            for gesture in inputmap.GESTURES[canvas]:
                row = self.mouse_table.rowCount()
                self.mouse_table.insertRow(row)
                self.mouse_table.setItem(row, 0, QTableWidgetItem(where))
                self.mouse_table.setItem(row, 1, QTableWidgetItem(keynames.gesture(gesture)))
                combo = QComboBox()
                combo.setObjectName("ent-input")
                for tool, text in inputmap.tools_for(canvas, gesture).items():
                    combo.addItem(text, tool)
                combo.currentIndexChanged.connect(
                    lambda _i, c=canvas, g=gesture, w=combo: self._set_mouse(c, g, w.currentData()))
                self.mouse_table.setCellWidget(row, 2, combo)
                self._mouse_combos[f"{canvas}:{gesture}"] = combo
        v.addWidget(self.mouse_table, 1)
        row = QHBoxLayout()
        reset = QPushButton("Reset the mouse")
        reset.setObjectName("tb-btn")
        reset.clicked.connect(self._reset_mouse)
        row.addWidget(reset)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _set_mouse(self, canvas: str, gesture: str, tool: str) -> None:
        key = f"{canvas}:{gesture}"
        if inputmap.DEFAULT_MOUSEMAP.get(key, "none") == tool:
            self._mousemap.pop(key, None)
        else:
            self._mousemap[key] = tool

    def _reset_mouse(self) -> None:
        self._mousemap = {}
        self._refresh_mouse()

    def _refresh_mouse(self) -> None:
        for key, combo in self._mouse_combos.items():
            canvas, gesture = key.split(":", 1)
            tool = inputmap.binding(canvas, gesture, self._mousemap)
            i = combo.findData(tool)
            combo.blockSignals(True)
            combo.setCurrentIndex(max(i, 0))
            combo.blockSignals(False)

    # -- load / save ---------------------------------------------------------

    def load(self, settings: VizSettings) -> None:
        self._keymap = {k: list(v) for k, v in settings.keymap.items()
                        if k in A.ACTION_BY_ID}
        self._mousemap = dict(settings.mousemap)
        self._refresh_keys()
        self._refresh_mouse()

    def apply_to(self, settings: VizSettings) -> None:
        settings.keymap = {k: list(v) for k, v in self._keymap.items()}
        settings.mousemap = dict(self._mousemap)

    def conflict_count(self) -> int:
        return len(A.conflicts(A.effective_keymap(self._keymap)))


def _palette_token(name: str, default: str) -> tuple[int, int, int, int]:
    """A palette token as RGBA (tokens may be ``rgba(...)``, which QColor
    cannot parse)."""
    from ...viz.theme import parse_colour

    try:
        from .. import theme_manager

        return parse_colour(theme_manager.CUR().get(name, default))
    except Exception:  # noqa: BLE001
        return parse_colour(default)


__all__ = ["ColourButton", "ShortcutsPage", "ViewerSettingsPage"]

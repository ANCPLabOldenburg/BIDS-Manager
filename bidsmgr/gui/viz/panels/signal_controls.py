"""The signal viewer's controls column, by purpose.

The toolbar holds what is reached for all the time: the amplitude, the
stretch of time on screen, the filter band, quality control, annotation and
the spectrum. Everything else is here, grouped by what it is FOR: which
channels and how they are drawn, moving through time, filtering and
resampling, events, quality control with its parameters, and the look.

The same sections, headers and restore buttons as the image viewer's
controls column (``panels.inspector``), so the two viewers read alike; the
presenter keeps owning the widgets (``sync_widgets`` updates them wherever
they are placed).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ....viz.settings import MeegQcSettings
from ...widgets.flow_layout import FlowBar
from .inspector import Section

log = logging.getLogger(__name__)


def _pills(widgets) -> FlowBar:
    """Buttons in a row that wraps rather than widening the column."""
    bar = FlowBar(h_spacing=6, v_spacing=6)
    for w in widgets:
        bar.addWidget(w)
    return bar


class _QcSection(Section):
    """Quality control: switch it on, read what it found, act on it, and set
    the check's parameters (generated from ``MeegQcSettings``)."""

    def restore_defaults(self) -> None:
        self.inspector.put_qc_settings(MeegQcSettings())
        self.inspector.apply_qc_settings()


class SignalControls(QWidget):
    """Sections for an MEG, EEG or physio recording."""

    def __init__(self, presenter) -> None:
        super().__init__()
        self.presenter = presenter
        self.ctx = presenter.ctx
        self.setObjectName("viz-inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._syncing = False
        self._qc_controls: dict[str, Any] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._sections: dict[str, Section] = {}
        for build in (self._channels, self._time, self._filters, self._events, self._qc,
                      self._display):
            section = build()
            self._sections[section.key] = section
            lay.addWidget(section)
        lay.addStretch(1)

    # -- plumbing ------------------------------------------------------------

    def run_guarded(self, fn: Callable[[Any], None], value: Any) -> None:
        if self._syncing or getattr(self.presenter, "_syncing", False):
            return
        try:
            fn(value)
        except ValueError as exc:
            self.presenter.viewer.status_message.emit(str(exc))

    def section(self, key: str) -> Section:
        return self._sections[key]

    def sections(self) -> list[str]:
        return list(self._sections)

    def _button(self, action_id: str) -> QPushButton:
        return self.presenter.viewer.action_manager.button(action_id)

    # -- sections ------------------------------------------------------------

    def _channels(self) -> Section:
        p = self.presenter
        s = Section(self, "signal.channels", "Channels")
        _combo, pick_button = p._widget_channels()
        s.add_row("type", "Type", p.type_combo,
                  "Which channel type is drawn. Types are never mixed on one scale: "
                  "magnetometers, gradiometers and EEG are in different units.")
        s.add_row("pick", None, pick_button, span=True)
        # Hidden together for a one-channel recording (with the toolbar's
        # Displayed channels, added when the toolbar is built).
        p._multi_channel_widgets = [p.type_combo, pick_button, s._rows["type"][0]]
        s.add_row("drawing", None, _pills(self._button(a) for a in (
            "traces.butterfly", "traces.normalize", "traces.page_scale", "traces.clip",
            "traces.dc")), span=True)
        return s

    def _time(self) -> Section:
        s = Section(self, "signal.time", "Time")
        s.add_row("moves", None, _pills(self._button(a) for a in (
            "time.fit", "traces.together")), span=True)
        return s

    def _filters(self) -> Section:
        p = self.presenter
        s = Section(self, "signal.filters", "Filters and resampling")
        p._widget_filter_fields()
        s.add_row("hp", "High-pass", p.hp_spin, p.hp_spin.toolTip())
        s.add_row("lp", "Low-pass", p.lp_spin, p.lp_spin.toolTip())
        s.add_row("notch", "Notch", p.notch_spin, p.notch_spin.toolTip())
        s.add_row("apply", None, _pills([p.filter_apply_button,
                                         self._button("traces.reset_filters")]), span=True)
        spin, button = p._widget_resample()
        s.add_row("resample", "Resample to", spin, spin.toolTip())
        s.add_row("resample_go", None, button, span=True)
        return s

    def _events(self) -> Section:
        p = self.presenter
        s = Section(self, "signal.events", "Events")
        s.add_row("toggle", None, _pills([self._button("events.toggle")]), span=True)
        s.add_row("source", "From", p._widget_events(),
                  "Where the events come from: the run's events.tsv, the trigger channel, "
                  "or the recording's annotations.")
        return s

    def _qc(self) -> Section:
        from ..settings_pages import _control_for

        p = self.presenter
        s = _QcSection(self, "signal.qc", "Quality control (QC)")
        s.add_row("toggle", None, _pills([self._button("traces.quality")]), span=True)
        from PyQt6.QtWidgets import QCheckBox

        self.qc_on_open = QCheckBox("Run QC when a file opens")
        self.qc_on_open.setChecked(self.ctx.settings.qc.on_open)
        self.qc_on_open.toggled.connect(
            lambda on: self.ctx.settings_hub.update(
                lambda st: setattr(st.qc, "on_open", bool(on))))
        s.add_row("on_open", None, self.qc_on_open,
                  "On: QC stays on from one recording to the next and runs as soon as one "
                  "opens. Off: every recording opens with QC off.", span=True)
        self.qc_summary = QLabel("Switch QC on to check every channel, type by type, and "
                                 "every segment of the recording.")
        self.qc_summary.setObjectName("sidecar-footer-summary")
        self.qc_summary.setWordWrap(True)
        s.add_row("summary", None, self.qc_summary, span=True)
        self.qc_report_button = QPushButton("Report...")
        self.qc_report_button.setObjectName("tb-btn")
        self.qc_report_button.setToolTip("Every channel and every flagged segment, with the "
                                         "reasons, and how to read them")
        self.qc_report_button.clicked.connect(p.show_quality_report)
        self.qc_mark_channels = QPushButton("Mark suggested channels bad")
        self.qc_mark_channels.setObjectName("tb-btn")
        self.qc_mark_channels.setToolTip("The noisy, flat and uncorrelated channels become "
                                         "bad channels: undoable, saved only with Save to "
                                         "dataset.")
        self.qc_mark_channels.clicked.connect(lambda _c=False: p.mark_channels())
        self.qc_mark_segments = QPushButton("Mark flagged segments bad")
        self.qc_mark_segments.setObjectName("tb-btn")
        self.qc_mark_segments.setToolTip("Every flagged segment becomes a bad segment, "
                                         "labelled by why (BAD_muscle, BAD_jump, BAD_noise).")
        self.qc_mark_segments.clicked.connect(lambda _c=False: p.mark_segments())
        s.add_row("act", None, _pills([self.qc_report_button, self.qc_mark_channels,
                                       self.qc_mark_segments]), span=True)
        heading = QLabel("Parameters")
        heading.setObjectName("viewer-meta-section")
        s.add_row("params", None, heading, span=True)
        # Which channel types QC runs on: one box per type in THIS recording.
        self.qc_types_bar = FlowBar(h_spacing=10, v_spacing=4)
        self._qc_type_boxes: dict[str, Any] = {}
        s.add_row("qc.types", "Channel types", self.qc_types_bar,
                  "The channel types QC runs on. Each is checked against its own type "
                  "only; untick a type to leave it out.")
        from PyQt6.QtWidgets import QAbstractSpinBox

        for name, info in MeegQcSettings.model_fields.items():
            control = _control_for(info)
            if control is None:
                continue
            if isinstance(control.widget, QAbstractSpinBox):
                # A decimal point whatever the system locale, like every
                # other number in the viewer ("0,010" reads as ten).
                control.widget.setLocale(p._c_locale())
            s.add_row(f"qc.{name}", info.title or name, control.widget,
                      info.description or "")
            self._qc_controls[name] = control
        self.qc_apply = QPushButton("Check again with these parameters")
        self.qc_apply.setObjectName("tb-btn-primary")
        self.qc_apply.setToolTip("Save the parameters (they are remembered) and run the "
                                 "check again.")
        self.qc_apply.clicked.connect(lambda: self.apply_qc_settings())
        s.add_row("qc.apply", None, self.qc_apply, span=True)
        self.put_qc_settings(self.ctx.settings.meeg_qc)
        return s

    def _display(self) -> Section:
        s = Section(self, "signal.display", "Display", open_=False)
        s.add_row("line", None, _pills([self._button("traces.line"),
                                        self._button("view.zen")]), span=True)
        return s

    # -- QC parameters ---------------------------------------------------------

    def put_qc_settings(self, settings: MeegQcSettings) -> None:
        self._syncing = True
        try:
            for name, control in self._qc_controls.items():
                control.put(getattr(settings, name))
            chosen = set(settings.types)
            for t, box in self._qc_type_boxes.items():
                box.setChecked(not chosen or t in chosen)
        finally:
            self._syncing = False

    def chosen_types(self) -> list[str]:
        """The ticked channel types; empty when every type is ticked."""
        ticked = [t for t, box in self._qc_type_boxes.items() if box.isChecked()]
        return [] if len(ticked) == len(self._qc_type_boxes) else ticked

    def _sync_type_boxes(self) -> None:
        """One box per data channel type of the recording on screen."""
        from PyQt6.QtWidgets import QCheckBox

        from ....viz.compute.meeg_qc import DATA_TYPES, type_label

        src = self.presenter.source
        # In the recording's channel order, as QC's own plots are.
        types = ([t for t in dict.fromkeys(getattr(src, "ch_types", None) or [])
                  if t in DATA_TYPES] if src is not None else [])
        if types == list(self._qc_type_boxes):
            return
        for box in self._qc_type_boxes.values():
            self.qc_types_bar.removeWidget(box)
            box.deleteLater()
        self._qc_type_boxes = {}
        chosen = set(self.ctx.settings.meeg_qc.types)
        self._syncing = True
        try:
            for t in types:
                box = QCheckBox(type_label(t))
                box.setChecked(not chosen or t in chosen)
                box.toggled.connect(lambda _on, ty=t: self._on_type_box(ty))
                self.qc_types_bar.addWidget(box)
                self._qc_type_boxes[t] = box
        finally:
            self._syncing = False

    def _on_type_box(self, ch_type: str) -> None:
        if self._syncing:
            return
        if not any(box.isChecked() for box in self._qc_type_boxes.values()):
            # At least one type: QC on nothing is not a check.
            box = self._qc_type_boxes[ch_type]
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
            return
        # Saved at once; QC runs again by itself if it is on (the parameters
        # it was computed with changed), and is not switched on by a tick.
        self.apply_qc_settings(run=False)

    def qc_settings(self) -> MeegQcSettings:
        """The parameters as the form shows them, on top of what is saved (the
        channel types are chosen elsewhere and must survive)."""
        values = self.ctx.settings.meeg_qc.model_dump()
        values.update({name: control.get() for name, control in self._qc_controls.items()})
        values["types"] = self.chosen_types()
        return MeegQcSettings(**values)

    def apply_qc_settings(self, *, run: bool = True) -> None:
        """Remember the parameters; with ``run``, check again (switching QC
        on: that is what the button asks for)."""
        try:
            new = self.qc_settings()
        except ValueError as exc:
            self.presenter.viewer.status_message.emit(str(exc))
            return
        self.ctx.settings_hub.update(lambda s: setattr(s, "meeg_qc", new))
        if run:
            self.presenter.recheck_quality()

    # -- state ---------------------------------------------------------------

    def sync(self) -> None:
        """Show what applies to this recording (QC and annotation are MEG and
        EEG only; 'all of this run' is physio only), and what QC found."""
        p = self.presenter
        meeg = p._file_kind == "meeg"
        if self.qc_on_open.isChecked() != self.ctx.settings.qc.on_open:
            self.qc_on_open.blockSignals(True)
            self.qc_on_open.setChecked(self.ctx.settings.qc.on_open)
            self.qc_on_open.blockSignals(False)
        qc = self._sections["signal.qc"]
        if qc.isHidden() == meeg:
            qc.setVisible(meeg)
        if meeg:
            self._sync_type_boxes()
        # Events only where the recording has a source of them (the label
        # "From" must not stand alone).
        src = p.source
        has_events = bool(src is not None and src.event_sources())
        events = self._sections["signal.events"]
        if events.isHidden() == has_events:
            events.setVisible(has_events)
        res = p.quality_result()
        on = bool(self.ctx.scene.traces.quality)
        if not on:
            text = ("Switch QC on to check every channel, type by type, and every segment "
                    "of the recording.")
        elif res is None:
            text = "Checking..."
        else:
            text = res["summary"]
        if self.qc_summary.text() != text:
            self.qc_summary.setText(text)
        ready = res is not None and on
        self.qc_report_button.setEnabled(ready)
        self.qc_mark_channels.setEnabled(bool(ready and res["suggested_bads"]))
        self.qc_mark_segments.setEnabled(bool(ready and any(res["flagged"])))


__all__ = ["SignalControls"]

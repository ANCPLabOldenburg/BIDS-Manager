"""The spectroscopy viewer's controls column, by purpose.

What is SHOWN (the spectrum or the FID, which component, which repeat and
edit condition), how it is PROCESSED for viewing (line broadening and
phase), the REFERENCE MARKS drawn on it (metabolite positions, the standard
window, the water reference), its QUALITY (signal-to-noise and NAA line
width, measured without line broadening), and the look. The same sections
as the image and signal viewers' columns.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ...widgets.flow_layout import FlowBar
from .inspector import Section


def _pills(widgets) -> FlowBar:
    bar = FlowBar(h_spacing=6, v_spacing=6)
    for w in widgets:
        bar.addWidget(w)
    return bar


class _ProcessingSection(Section):
    def restore_defaults(self) -> None:
        self.inspector.presenter._run_ui("spectrum.reset_processing")


class SpectrumControls(QWidget):
    """Sections for an MR spectrum."""

    def __init__(self, presenter) -> None:
        super().__init__()
        self.presenter = presenter
        self.ctx = presenter.ctx
        self.setObjectName("viz-inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        p = presenter
        p._make_processing()
        tips = p.tips
        self._sections: dict[str, Section] = {}

        shown = Section(self, "spectrum.shown", "What is shown")
        shown.add_row("domain", "Domain", p.domain, tips["domain"])
        shown.add_row("part", "Component", p.part, tips["part"])
        shown.add_row("repeat", "Repeats", p.repeat, tips["repeat"])
        shown.add_row("edit", "Edit condition", p.edit, tips["edit"])
        self._add(lay, shown)

        processing = _ProcessingSection(self, "spectrum.processing", "Processing for viewing")
        processing.add_row("lb", "Line broadening", p.lb, tips["lb"])
        processing.add_row("phase0", "Zero-order phase", p.phase0, tips["phase0"])
        processing.add_row("phase1", "First-order phase", p.phase1, tips["phase1"])
        processing.add_row("phase_auto", None, _pills([self._button("spectrum.auto_phase"),
                                                       self._button("spectrum.reset_processing")]),
                           span=True)
        self._add(lay, processing)

        marks = Section(self, "spectrum.marks", "Reference marks")
        marks.add_row("marks", None, _pills(self._button(a) for a in (
            "spectrum.metabolites", "spectrum.window", "spectrum.reference",
            "spectrum.water")), span=True)
        self._add(lay, marks)

        qc = Section(self, "spectrum.qc", "Quality control (QC)")
        self.qc_text = QLabel("")
        self.qc_text.setObjectName("sidecar-footer-summary")
        self.qc_text.setWordWrap(True)
        self.qc_text.setToolTip(
            "Signal-to-noise and the NAA line width, measured on the spectrum as phased "
            "and WITHOUT line broadening (broadening improves the one and worsens the "
            "other). A wide line or a low SNR says to look at the shim and the repeats.")
        qc.add_row("metrics", None, self.qc_text, span=True)
        self._add(lay, qc)

        display = Section(self, "spectrum.display", "Display", open_=False)
        display.add_row("view", None, _pills(self._button(a) for a in (
            "spectrum.fit", "spectrum.reset", "traces.line")), span=True)
        self._add(lay, display)
        lay.addStretch(1)

    def _add(self, lay, section: Section) -> None:
        self._sections[section.key] = section
        lay.addWidget(section)

    def _button(self, action_id: str) -> QPushButton:
        return self.presenter.viewer.action_manager.button(action_id)

    def run_guarded(self, fn, value) -> None:
        fn(value)

    def section(self, key: str) -> Section:
        return self._sections[key]

    def sections(self) -> list[str]:
        return list(self._sections)

    def sync(self, edits: int) -> None:
        self._sections["spectrum.shown"].show_row("edit", edits > 1)
        text = self.presenter.quality() or "Open a spectrum to measure it."
        if self.qc_text.text() != text:
            self.qc_text.setText(text)


__all__ = ["SpectrumControls"]

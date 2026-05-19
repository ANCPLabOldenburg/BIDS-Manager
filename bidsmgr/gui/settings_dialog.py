"""Settings dialog — surface CLI knobs the GUI uses.

Reads / writes :class:`bidsmgr.gui.app_settings.AppSettings` via
``QSettings``. All changes are applied on **Save** (no live binding) so
the user can experiment with values and cancel without commit.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .app_settings import AppSettings


class SettingsDialog(QDialog):
    """Three-tab settings dialog: Display / Scan / Convert.

    Theme + post-convert chain live under their natural homes. The
    inspector column visibility is NOT here — it's controlled via the
    table header's right-click menu, and that menu writes through to
    the same QSettings namespace.
    """

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager — Settings")
        self.resize(540, 560)
        self._settings = settings

        v = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_display_tab(), "Display")
        tabs.addTab(self._build_scan_tab(), "Scan")
        tabs.addTab(self._build_convert_tab(), "Convert + post-convert")
        v.addWidget(tabs, 1)

        # Save / Cancel.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    # Font scale presets shown in the Display tab. The combo stores the
    # human-readable label; the float multiplier is the second element.
    _FONT_SCALE_PRESETS: list[tuple[str, float]] = [
        ("Compact (0.85x)",        0.85),
        ("Normal (1.00x)",         1.00),
        ("Comfortable (1.15x)",    1.15),
        ("Large (1.30x)",          1.30),
        ("Extra large (1.50x)",    1.50),
    ]

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        self._theme_combo.setCurrentText(self._settings.theme)
        form.addRow("Theme:", self._theme_combo)

        # Font scale: multiplies every font-size (QSS + delegate paints +
        # inline stylesheets + icon sizes) so the user can comfortably
        # nudge the whole UI up or down without touching individual
        # widgets. The user's choice is persisted in QSettings under
        # ``ui/font_scale``.
        self._font_scale_combo = QComboBox()
        for label, _value in self._FONT_SCALE_PRESETS:
            self._font_scale_combo.addItem(label)
        current_idx = self._closest_font_scale_index(self._settings.font_scale)
        self._font_scale_combo.setCurrentIndex(current_idx)
        form.addRow("Font scale:", self._font_scale_combo)

        # Header brand artwork. "Default" = minimalist mark inverted on
        # dark; "App icon" = full-color BIDS-Manager application icon.
        self._header_logo_combo = QComboBox()
        for label, _value in self._HEADER_LOGO_PRESETS:
            self._header_logo_combo.addItem(label)
        self._header_logo_combo.setCurrentIndex(
            self._header_logo_index(self._settings.header_logo)
        )
        form.addRow("Header logo:", self._header_logo_combo)

        hint = QLabel(
            "Theme can also be toggled live via the sun / moon button "
            "in the top header. Font scale and header logo apply on Save."
        )
        hint.setStyleSheet("color: #8b949e;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        return w

    # Combo presets for the header brand mark.
    _HEADER_LOGO_PRESETS: list[tuple[str, str]] = [
        ("Default (monochrome mark)", "default"),
        ("App icon (full color)",     "app_icon"),
    ]

    @classmethod
    def _header_logo_index(cls, value: str) -> int:
        for i, (_label, key) in enumerate(cls._HEADER_LOGO_PRESETS):
            if key == value:
                return i
        return 0

    @classmethod
    def _closest_font_scale_index(cls, value: float) -> int:
        """Return the preset index whose multiplier is nearest *value*.

        Lets us round-trip an arbitrary persisted float (e.g. a hand-
        edited QSettings value) back to the nearest combo entry without
        rejecting it.
        """
        try:
            return min(
                range(len(cls._FONT_SCALE_PRESETS)),
                key=lambda i: abs(cls._FONT_SCALE_PRESETS[i][1] - value),
            )
        except Exception:
            return 1  # "Normal"

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Defaults group.
        defaults = QGroupBox("Scan defaults")
        form = QFormLayout(defaults)

        self._scan_jobs = QSpinBox()
        self._scan_jobs.setRange(1, 64)
        self._scan_jobs.setValue(self._settings.scan_n_jobs)
        form.addRow("Parallel workers (-j):", self._scan_jobs)

        self._scan_dataset = QLineEdit()
        self._scan_dataset.setText(self._settings.dataset_slug)
        self._scan_dataset.setPlaceholderText("(auto-derive from raw folder name)")
        form.addRow("Default dataset slug:", self._scan_dataset)

        self._scan_line_freq = QDoubleSpinBox()
        self._scan_line_freq.setRange(0.0, 100.0)
        self._scan_line_freq.setDecimals(1)
        self._scan_line_freq.setSingleStep(1.0)
        self._scan_line_freq.setValue(self._settings.scan_line_freq)
        form.addRow("EEG/MEG line frequency (Hz):", self._scan_line_freq)

        self._scan_montage = QLineEdit()
        self._scan_montage.setText(self._settings.scan_montage)
        self._scan_montage.setPlaceholderText("e.g. standard_1005, biosemi64")
        form.addRow("EEG/MEG montage:", self._scan_montage)

        self._scan_probe = QCheckBox(
            "Enable --probe-convert (run dcm2niix per series to enrich "
            "naming with the actual file count + extensions)"
        )
        self._scan_probe.setChecked(self._settings.scan_probe_convert)
        form.addRow("Probe:", self._scan_probe)

        self._scan_skip_bids_guess = QCheckBox(
            "Skip dcm2niix BidsGuess classifier (use only the legacy "
            "regex fallback layer)"
        )
        self._scan_skip_bids_guess.setChecked(self._settings.scan_skip_bids_guess)
        form.addRow("Classifier:", self._scan_skip_bids_guess)

        v.addWidget(defaults)
        v.addStretch(1)
        return w

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        convert = QGroupBox("Convert defaults")
        form = QFormLayout(convert)
        self._convert_jobs = QSpinBox()
        self._convert_jobs.setRange(1, 64)
        self._convert_jobs.setValue(self._settings.convert_n_jobs)
        form.addRow("Parallel workers (-j):", self._convert_jobs)

        self._convert_overwrite = QCheckBox(
            "Overwrite existing subjects in the BIDS output (otherwise "
            "the convert verb refuses to clobber)"
        )
        self._convert_overwrite.setChecked(self._settings.convert_overwrite)
        form.addRow("Overwrite:", self._convert_overwrite)

        v.addWidget(convert)

        # Post-convert chain.
        post = QGroupBox("Post-convert chain (run after every conversion)")
        pform = QFormLayout(post)

        self._post_run_metadata = QCheckBox(
            "Run bidsmgr-metadata (writes dataset_description, "
            "participants.tsv, *_scans.tsv, sidecar audit)"
        )
        self._post_run_metadata.setChecked(self._settings.post_run_metadata)
        pform.addRow(self._post_run_metadata)

        self._post_metadata_fill_todos = QCheckBox(
            "  └ insert 'TODO' placeholders for missing recommended fields"
        )
        self._post_metadata_fill_todos.setChecked(self._settings.post_metadata_fill_todos)
        pform.addRow(self._post_metadata_fill_todos)

        self._post_run_validate = QCheckBox(
            "Run bidsmgr-validate (schema + bidsschematools structural)"
        )
        self._post_run_validate.setChecked(self._settings.post_run_validate)
        pform.addRow(self._post_run_validate)

        self._post_validate_strict = QCheckBox(
            "  └ --strict (warnings → errors)"
        )
        self._post_validate_strict.setChecked(self._settings.post_validate_strict)
        pform.addRow(self._post_validate_strict)

        self._post_validate_html = QCheckBox(
            "  └ --html (write self-contained validation_report.html)"
        )
        self._post_validate_html.setChecked(self._settings.post_validate_html)
        pform.addRow(self._post_validate_html)

        v.addWidget(post)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        s = self._settings
        s.theme = self._theme_combo.currentText()
        s.font_scale = self._FONT_SCALE_PRESETS[
            self._font_scale_combo.currentIndex()
        ][1]
        s.header_logo = self._HEADER_LOGO_PRESETS[
            self._header_logo_combo.currentIndex()
        ][1]

        s.scan_n_jobs = self._scan_jobs.value()
        s.dataset_slug = self._scan_dataset.text().strip()
        s.scan_line_freq = self._scan_line_freq.value()
        s.scan_montage = self._scan_montage.text().strip()
        s.scan_probe_convert = self._scan_probe.isChecked()
        s.scan_skip_bids_guess = self._scan_skip_bids_guess.isChecked()

        s.convert_n_jobs = self._convert_jobs.value()
        s.convert_overwrite = self._convert_overwrite.isChecked()

        s.post_run_metadata = self._post_run_metadata.isChecked()
        s.post_metadata_fill_todos = self._post_metadata_fill_todos.isChecked()
        s.post_run_validate = self._post_run_validate.isChecked()
        s.post_validate_strict = self._post_validate_strict.isChecked()
        s.post_validate_html = self._post_validate_html.isChecked()

        s.save()
        self.accept()


__all__ = ["SettingsDialog"]

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

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        self._theme_combo.setCurrentText(self._settings.theme)
        form.addRow("Theme:", self._theme_combo)

        hint = QLabel(
            "Theme can also be toggled live via the ☀/☾ button in the top header."
        )
        hint.setStyleSheet("color: #8b949e;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        return w

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

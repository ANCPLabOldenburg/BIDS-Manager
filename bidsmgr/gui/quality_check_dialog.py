"""Tools -> Quality check: every anatomical and diffusion image of the
dataset, checked (``bidsmgr.qc``), one row each.

A table per kind of image, because an anatomical and a diffusion image are
judged by different measures. Each value is coloured by where it sits among
the dataset's other images of the same suffix and acquisition (a robust z:
one bad image does not move the yardstick), so the image to look at first is
the one with colour in its row. Double-click a row: the image opens in the
viewer with its full report, the evidence a click away.

Results are written into ``derivatives/bidsmgr-qc/`` as they come, so the
dialog opens on what was checked before, and checking again only what is
new is the default. Non-modal: the images are looked at while it stays open.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..qc import report as RP
from ..qc import run as RUN
from .dialog_chrome import build_footer_with, build_header, hint
from .widgets.spinner import BusySpinner
from .typefaces import code

#: The measures a row shows, per kind: (key, header, full name).
ANAT_COLUMNS = (
    ("snr_wm", "SNR (WM)", "SNR in white matter"),
    ("cjv", "Joint variation", "Coefficient of joint variation of grey and white matter"),
    ("cnr", "Contrast/noise", "Contrast-to-noise ratio (needs real air)"),
    ("efc", "Entropy focus", "Entropy focus criterion: ghosting and blurring (needs real air)"),
    ("qi1", "Air artefacts %", "Share of the air holding structured signal (needs real air)"),
    ("fwhm_avg", "Smoothness mm", "Smoothness, mean of the three axes"),
    ("inu_range", "Non-uniformity", "Spread of the bias field over the brain"),
    ("fov_cut", "Brain cut %", "Share of the brain outside the field of view"),
)
DWI_COLUMNS = (
    ("motion_b0", "Motion mm", "Head motion across the b=0 volumes"),
    ("volumes_far", "Far volumes", "Volumes far out of place"),
    ("dropout_slices", "Dropouts", "Slices with signal dropout"),
    ("spikes_ppm", "Spikes ppm", "Spiking voxels per million"),
    ("snr_cc_b0", "SNR (CC, b=0)", "SNR in the corpus callosum at b=0"),
    ("ndc", "Neighbour corr.", "Correlation of each volume with its nearest direction"),
    ("drift", "Drift %", "Signal drift over the scan"),
)
_FINDING_ORDER = {"error": 0, "warning": 1, "info": 2, "ok": 3}


def _item(text: str, *, sort=None, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if sort is not None:
        item.setData(Qt.ItemDataRole.UserRole, sort)
    if tip:
        item.setToolTip(tip)
    return item


class _SortItem(QTableWidgetItem):
    """Sorts by its number, not its text."""

    def __lt__(self, other) -> bool:  # noqa: D105
        a = self.data(Qt.ItemDataRole.UserRole)
        b = other.data(Qt.ItemDataRole.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        if isinstance(a, (int, float)):
            return True
        return super().__lt__(other)


class QualityCheckDialog(QDialog):
    """Check the dataset's images and read the results as a table."""

    def __init__(self, root: Path, *, open_image: Optional[Callable[[Path], None]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._open_image = open_image
        self._worker = None
        self._rows: list[dict] = []
        self.setWindowTitle("Quality check")
        self.setModal(False)
        self.resize(1320, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "How good are the images?",
            "A fast quality check of every anatomical and diffusion image: noise, "
            "contrast, artefacts, coverage, the gradient table, motion and slice "
            "dropout. <b>The air around the head is measured only in images that "
            "are not defaced.</b> Results go into " + code("derivatives/bidsmgr-qc/") + "; "
            "coloured values sit far from the dataset's other images of the same kind."))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.run_button = QPushButton("Check")
        self.run_button.setObjectName("tb-btn-primary")
        self.run_button.setToolTip("Check the images: only those not checked yet, unless "
                                   "Check again is on")
        self.run_button.clicked.connect(self.start)
        bar.addWidget(self.run_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("tb-btn")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop after the images already running")
        self.stop_button.clicked.connect(self.stop)
        bar.addWidget(self.stop_button)
        self.again_box = QCheckBox("Check again")
        self.again_box.setToolTip("Check every image, also those already checked")
        bar.addWidget(self.again_box)
        jobs_label = QLabel("At once")
        jobs_label.setObjectName("sidecar-footer-summary")
        bar.addWidget(jobs_label)
        self.jobs = QSpinBox()
        self.jobs.setRange(1, max(1, os.cpu_count() or 1))
        self.jobs.setValue(min(4, max(1, (os.cpu_count() or 2) // 2)))
        self.jobs.setToolTip("Images checked at the same time, each in its own process")
        bar.addWidget(self.jobs)
        from ..viz.settings import qc_config
        from .viz.bridge import SettingsHub

        settings_cfg = qc_config(SettingsHub.instance().settings)
        self.flips_box = QCheckBox("Look for flipped b-vectors")
        self.flips_box.setChecked(settings_cfg.dwi.check_flips)
        self.flips_box.setToolTip("Experimental: fit the tensor with every sign flip and axis "
                                  "swap of the table and keep the most coherent (a few "
                                  "seconds per diffusion image)")
        bar.addWidget(self.flips_box)
        self.fast_box = QCheckBox("Fast masks")
        self.fast_box.setToolTip("For this run, approximate brain and tissue masks from the "
                                 "fast methods instead of those chosen in Settings > Quality "
                                 "control (mindgrab, the tissue model and niimath by "
                                 "default): about 15 s faster per anatomical image, less "
                                 "accurate. Every threshold comes from Settings.")
        bar.addWidget(self.fast_box)
        bar.addStretch(1)
        self.spinner = BusySpinner()
        bar.addWidget(self.spinner)
        bl.addLayout(bar)

        self.tabs = QTabWidget()
        self.anat_table = self._table(ANAT_COLUMNS, "anat")
        self.dwi_table = self._table(DWI_COLUMNS, "dwi")
        self.tabs.addTab(self.anat_table, "Anatomical")
        self.tabs.addTab(self.dwi_table, "Diffusion")
        bl.addWidget(self.tabs, 1)
        self.details = hint("Select an image to read its findings; double-click it to open "
                            "it in the viewer with its full report.")
        bl.addWidget(self.details)
        outer.addWidget(body, 1)

        buttons = QDialogButtonBox()
        close = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close.setObjectName("tb-btn-primary")
        buttons.rejected.connect(self.close)
        self.status = QLabel("")
        self.status.setObjectName("sidecar-footer-summary")
        outer.addWidget(build_footer_with(self.status, buttons))

        self.candidates = RUN.find_images(self._root)
        self.load_saved()

    # -- tables -----------------------------------------------------------------

    def _table(self, columns, kind: str) -> QTableWidget:
        from ..qc import explain

        table = QTableWidget(0, 3 + len(columns), self)
        table.setHorizontalHeaderLabels(["Image", "Suffix", "Findings"]
                                        + [c[1] for c in columns])
        for c, (key, _h, full) in enumerate(columns, start=3):
            # The measure, what it is, and which way is better: the same
            # words as the viewer's quality panel.
            exp = explain.lookup(f"{kind}.{key}")
            table.horizontalHeaderItem(c).setToolTip(
                f"{full}. {exp.short}" if exp is not None else full)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        # One line a row: a path wrapped over two lines doubled every row.
        table.setWordWrap(False)
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for c in range(1, table.columnCount()):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        table.itemSelectionChanged.connect(lambda t=table: self._describe(t))
        table.cellDoubleClicked.connect(lambda r, _c, t=table: self._open(t, r))
        table.setProperty("columns", [c[0] for c in columns])
        return table

    def load_saved(self) -> None:
        """Show what was checked before (the derivative)."""
        self._rows = RP.load_all(self._root)
        self._fill()
        done = {str(r.get("path")) for r in self._rows}
        left = [p for p in self.candidates
                if p.relative_to(self._root).as_posix() not in done]
        self.status.setText(
            f"{len(self.candidates)} images in the dataset, {len(self._rows)} checked"
            + (f", {len(left)} not yet" if left else "") + ".")

    def _fill(self) -> None:
        from .viz.bridge import ThemeHub

        theme = ThemeHub.instance().theme
        warn = QColor(theme.token("warning", "#d29922"))
        zs = RP.group_z(self._rows)
        for table, kind, columns in ((self.anat_table, "anat", ANAT_COLUMNS),
                                     (self.dwi_table, "dwi", DWI_COLUMNS)):
            rows = [r for r in self._rows if r.get("kind") == kind]
            table.setSortingEnabled(False)
            table.setRowCount(len(rows))
            for i, r in enumerate(sorted(rows, key=lambda x: str(x.get("path", "")))):
                path = str(r.get("path", ""))
                # The file's name (its entities say whose and which); the
                # folders are in the tooltip.
                shown = Path(path).name.replace(".nii.gz", "").replace(".nii", "")
                name = _item(shown, tip=path)
                name.setData(Qt.ItemDataRole.UserRole, path)
                table.setItem(i, 0, name)
                table.setItem(i, 1, _item(str(r.get("suffix", ""))))
                serious = [f for f in r.get("findings", [])
                           if f.get("level") in ("warning", "error")]
                errors = sum(1 for f in serious if f["level"] == "error")
                text = (f"{errors} error{'s' if errors != 1 else ''}, " if errors else "") + (
                    f"{len(serious) - errors} warning{'s' if len(serious) - errors != 1 else ''}"
                    if len(serious) - errors else "")
                f_item = _SortItem(text.strip(", ") or "none")
                f_item.setData(Qt.ItemDataRole.UserRole, float(len(serious) + 10 * errors))
                f_item.setToolTip("\n".join(f"{f['title']}: {f['message']}" for f in serious))
                if serious:
                    f_item.setForeground(warn)
                table.setItem(i, 2, f_item)
                info = r.get("metric_info", {})
                z_of = zs.get(path, {})
                for c, (key, _h, full) in enumerate(columns, start=3):
                    value = r.get("metrics", {}).get(key)
                    meta = info.get(key, {})
                    if value is None:
                        item = _SortItem("n/a")
                        item.setToolTip(meta.get("why_missing", "") or full)
                        item.setForeground(QColor(theme.dim))
                    else:
                        fmt = meta.get("fmt", "{:.3g}")
                        try:
                            shown = fmt.format(value)
                        except (ValueError, TypeError):
                            shown = f"{value:.3g}"
                        item = _SortItem(shown)
                        item.setData(Qt.ItemDataRole.UserRole, float(value))
                        z = z_of.get(key)
                        better = meta.get("better", "")
                        tip = full
                        if z is not None:
                            tip += f"\nRobust z among comparable images: {z:+.1f}"
                            worse = (better == "higher" and z < 0) or (better == "lower" and z > 0)
                            if worse and abs(z) >= RP.GROUP_Z:
                                item.setForeground(warn)
                                font = item.font()
                                font.setBold(True)
                                item.setFont(font)
                        item.setToolTip(tip)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                    table.setItem(i, c, item)
            table.setSortingEnabled(True)
            fm = table.fontMetrics()
            widest = max([fm.horizontalAdvance(table.item(k, 0).text())
                          for k in range(table.rowCount())] or [200])
            table.setColumnWidth(0, min(max(widest + 28, 220), 460))
        self.tabs.setTabText(0, f"Anatomical ({self.anat_table.rowCount()})")
        self.tabs.setTabText(1, f"Diffusion ({self.dwi_table.rowCount()})")

    def _row_of(self, table: QTableWidget, r: int) -> Optional[dict]:
        item = table.item(r, 0)
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return next((x for x in self._rows if str(x.get("path")) == path), None)

    def _describe(self, table: QTableWidget) -> None:
        rows = table.selectionModel().selectedRows()
        if not rows:
            return
        r = self._row_of(table, rows[0].row())
        if r is None:
            return
        found = sorted(r.get("findings", []), key=lambda f: _FINDING_ORDER.get(f["level"], 9))
        if not found:
            self.details.setText(f"<b>{r['path']}</b>: nothing to flag.")
            return
        lines = "".join(f"<br><b>{f['title']}</b> ({f['level']}): {f['message']}"
                        for f in found[:6])
        self.details.setText(f"<b>{r['path']}</b>{lines}")

    def _open(self, table: QTableWidget, r: int) -> None:
        row = self._row_of(table, r)
        if row is not None and self._open_image is not None:
            self._open_image(self._root / str(row["path"]))

    # -- running ----------------------------------------------------------------

    def start(self) -> None:
        from ..workers.quality import QualityWorker

        if self._worker is not None:
            return
        done = {str(r.get("path")) for r in self._rows}
        paths = (list(self.candidates) if self.again_box.isChecked() else
                 [p for p in self.candidates
                  if p.relative_to(self._root).as_posix() not in done])
        if not paths:
            self.status.setText("Every image is checked. Turn on Check again to check them "
                                "again.")
            return
        worker = QualityWorker(self._root, paths, jobs=self.jobs.value(),
                               config=self.config(), parent=self)
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.spinner.set_busy(True, message=f"Checking {len(paths)} images")
        self.status.setText(f"Checking {len(paths)} images...")
        worker.start()

    def config(self):
        """The configuration this run uses: Settings > Quality control, with
        this dialog's two switches on top."""
        from ..qc.config import fast
        from ..viz.settings import qc_config
        from .viz.bridge import SettingsHub

        cfg = qc_config(SettingsHub.instance().settings).model_copy(deep=True)
        cfg.dwi.check_flips = self.flips_box.isChecked()
        return fast(cfg) if self.fast_box.isChecked() else cfg

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the images already running...")

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.status.setText(f"Checked {done} of {total}: {name}")
        self.spinner.set_busy(True, message=f"{done} of {total}")

    def _finish(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait(2000)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.spinner.set_busy(False, message="")

    def _on_done(self, _rows) -> None:
        self._finish()
        self.load_saved()

    def _on_failed(self, message: str) -> None:
        self._finish()
        self.status.setText(f"The check stopped: {message.splitlines()[0]}")
        self.load_saved()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(30000)
            self._worker = None
        super().closeEvent(event)


__all__ = ["ANAT_COLUMNS", "DWI_COLUMNS", "QualityCheckDialog"]

"""The preview window stays in front on macOS, also while it is used.

Two causes found on 2026-10-19, macOS only:
- widgets shown before they had a parent became windows for a moment and
  took the focus (the shortcuts list's cards; earlier the spinners and the
  viewer's side tab), and when such a window went, macOS handed the focus to
  the main window, which came to the front over the preview;
- a dialog is a panel on macOS, which can never be the app's main window, so
  whenever something inside it closed, the focus went to the main window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QDialog, QMainWindow, QMenu

pytestmark = pytest.mark.gui

#: Actions that would write, pick a file or leave the window: not for a sweep.
_SKIP = ("save", "export", "quit", "close", "open ", "load", "browse", "choose",
         "pick", "edit", "copy")


class _Strays(QObject):
    """Every window shown that is not meant to be one."""

    def __init__(self) -> None:
        super().__init__()
        self.current = ""
        self.found: dict[str, set] = {}

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if (event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow()
                and not isinstance(obj, (QDialog, QMenu, QMainWindow))
                and type(obj).__name__ not in ("QTipLabel", "QComboBoxPrivateContainer")):
            self.found.setdefault(type(obj).__name__, set()).add(self.current)
        return False


@pytest.fixture
def preview(qtbot, tmp_path: Path):
    nib = pytest.importorskip("nibabel")
    from bidsmgr.gui.preview_dialog import PreviewDialog

    image = tmp_path / "sub-01_task-rest_bold.nii.gz"
    data = (np.random.default_rng(0).random((24, 24, 16, 6)) * 100).astype("float32")
    nib.save(nib.Nifti1Image(data, np.eye(4)), image)
    (tmp_path / "sub-01_task-rest_bold.json").write_text('{"RepetitionTime": 2.0}')
    dlg = PreviewDialog("rest", tmp_path)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.set_files([image], tmp_path)
    qtbot.waitUntil(lambda: dlg.viewer is not None and dlg.viewer.is_loaded(), timeout=20_000)
    return dlg


def test_no_viewer_action_in_a_preview_opens_a_stray_window(qtbot, qapp, preview) -> None:
    from PyQt6.QtCore import QTimer

    spy = _Strays()
    qapp.installEventFilter(spy)

    def close_modal() -> None:
        modal = qapp.activeModalWidget()
        if modal is not None:
            modal.reject() if hasattr(modal, "reject") else modal.close()

    timer = QTimer()
    timer.timeout.connect(close_modal)
    timer.start(100)
    try:
        actions = [a for a in preview.findChildren(QAction) if a.text().strip() and not a.menu()]
        assert len(actions) > 50, "the sweep must reach the viewer's actions"
        for action in actions:
            text = action.text().replace("&", "")
            if any(word in text.lower() for word in _SKIP):
                continue
            spy.current = text
            action.trigger()
            qapp.processEvents()
            for w in qapp.topLevelWidgets():
                if w.isVisible() and w is not preview and isinstance(w, QDialog):
                    w.close()
    finally:
        timer.stop()
        qapp.removeEventFilter(spy)
    assert spy.found == {}, f"windows that should not exist, and what opened them: {spy.found}"


@pytest.mark.parametrize(("platform", "kind"), [("darwin", Qt.WindowType.Window),
                                                ("linux", Qt.WindowType.Dialog),
                                                ("win32", Qt.WindowType.Dialog)])
def test_the_preview_is_a_normal_window_on_macos(qtbot, monkeypatch, platform, kind) -> None:
    import bidsmgr.gui.preview_dialog as module

    monkeypatch.setattr(module.sys, "platform", platform)
    dlg = module.PreviewDialog("rest", None)
    qtbot.addWidget(dlg)
    assert dlg.windowType() == kind


def test_a_spinner_made_busy_before_it_is_placed_waits_for_its_parent(qtbot, qapp) -> None:
    """Shown without a parent, a spinner is a window: on macOS it took the
    focus from the preview when the quality check started (2026-10-19)."""
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from bidsmgr.gui.widgets.spinner import BusySpinner

    spy = _Strays()
    qapp.installEventFilter(spy)
    try:
        spinner = BusySpinner()
        spinner.set_busy(True, message="Checking")
        qapp.processEvents()
        assert not spinner.isVisible() and spinner.is_busy()
        host = QWidget()
        qtbot.addWidget(host)
        QVBoxLayout(host).addWidget(spinner)
        host.show()
        qtbot.waitUntil(spinner.isVisible, timeout=2_000)
        assert not spinner.isWindow()
        spinner.set_busy(False)
        assert not spinner.isVisible()
    finally:
        qapp.removeEventFilter(spy)
    assert "BusySpinner" not in spy.found, "the spinner was a window for a moment"


@pytest.fixture
def anatomy_preview(qtbot, tmp_path: Path, monkeypatch):
    """A preview of a T1w: a bright ellipsoid head in dark air."""
    nib = pytest.importorskip("nibabel")
    from bidsmgr.gui.preview_dialog import PreviewDialog

    monkeypatch.setenv("BIDSMGR_QC_ENGINE", "numpy")     # no model run in a test
    shape = (48, 56, 48)
    grid = np.indices(shape).astype("float32")
    centre = np.array(shape, dtype="float32")[:, None, None, None] / 2
    radius = np.array([18, 22, 18], dtype="float32")[:, None, None, None]
    inside = (((grid - centre) / radius) ** 2).sum(axis=0) < 1
    rng = np.random.default_rng(1)
    data = rng.normal(8, 3, shape).clip(0) + inside * rng.normal(600, 40, shape)
    image = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(data.astype("float32"), np.diag([1.0, 1.0, 1.0, 1.0])), image)
    (tmp_path / "sub-01_T1w.json").write_text('{"MagneticFieldStrength": 3}')
    dlg = PreviewDialog("anat", tmp_path)
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.set_files([image], tmp_path)
    qtbot.waitUntil(lambda: dlg.viewer is not None and dlg.viewer.is_loaded(), timeout=20_000)
    return dlg


def test_the_quality_check_of_an_anatomy_opens_no_stray_window(qtbot, qapp,
                                                                anatomy_preview) -> None:
    from PyQt6.QtWidgets import QCheckBox

    presenter = anatomy_preview.viewer.presenter
    spy = _Strays()
    qapp.installEventFilter(spy)
    try:
        spy.current = "check image quality"
        anatomy_preview.viewer.trigger("qc.check")
        qtbot.waitUntil(lambda: presenter._quality_result is not None, timeout=60_000)
        boxes = presenter._quality_panel.findChildren(QCheckBox)
        assert boxes, "the result offers its maps"
        for box in boxes:
            spy.current = f"map {box.text()}"
            box.click()
            qapp.processEvents()
        spy.current = "check again"
        presenter._quality_panel.check_requested.emit()
        qapp.processEvents()
        qtbot.waitUntil(lambda: presenter._quality_result is not None
                        and not presenter.ctx.jobs.running("quality"), timeout=60_000)
    finally:
        qapp.removeEventFilter(spy)
    assert spy.found == {}, f"windows that should not exist, and what opened them: {spy.found}"

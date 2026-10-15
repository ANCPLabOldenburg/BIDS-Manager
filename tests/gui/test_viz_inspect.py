"""The header inspector in the viewer, and a validation finding opening it.

The rows themselves are tested Qt-free in ``tests/unit/test_viz_inspect.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.gui.viz import Viewer  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    """A run whose header says 2.5 s and whose sidecar says 2 s."""
    root = tmp_path / "Study"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    img = nib.Nifti1Image(np.zeros((4, 4, 6, 5), np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1.0, 1.0, 1.0, 2.5))
    nib.save(img, str(func / "sub-01_task-x_bold.nii.gz"))
    (func / "sub-01_task-x_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "x", "BIDSVersion": "1.10.0"}))
    return root


def _bold(root: Path) -> Path:
    return root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"


def _viewer(qtbot) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(900, 600)
    v.show()
    qtbot.waitExposed(v)
    return v


class TestTheInspector:
    def test_it_opens_on_the_image_and_says_what_disagrees(self, qtbot, ds):
        v = _viewer(qtbot)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(_bold(ds), ds)
        dlg = v.show_header()
        qtbot.addWidget(dlg)
        assert dlg.status() == "error"
        rows = {field: (head, side, status) for field, head, side, status in dlg.row_texts()}
        assert rows["Repetition time"] == ("2.5 s", "2 s", "error")
        assert "1 disagreement" in dlg.summary.text()

    def test_a_rule_selects_its_row(self, qtbot, ds):
        v = _viewer(qtbot)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(_bold(ds), ds)
        dlg = v.show_header("REPETITION_TIME_MISMATCH")
        qtbot.addWidget(dlg)
        assert dlg.tree.currentItem().text(0).strip() == "Repetition time"

    def test_asked_for_while_opening_it_follows(self, qtbot, ds):
        v = _viewer(qtbot)
        v.set_file(_bold(ds), ds)
        assert v.show_header("REPETITION_TIME_MISMATCH") is None
        qtbot.waitUntil(lambda: v.presenter.header_dialog is not None, timeout=20_000)
        dlg = v.presenter.header_dialog
        qtbot.addWidget(dlg)
        assert dlg.tree.currentItem().text(0).strip() == "Repetition time"

    def test_it_is_in_the_tools_menu_and_on_a_key(self, qtbot, ds):
        v = _viewer(qtbot)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(_bold(ds), ds)
        texts = [a.text() for a in v.presenter.tools_button.menu().actions()]
        assert "Header and sidecar..." in texts
        assert v.action("header.show").shortcut().toString() == "H"

    def test_opening_it_again_replaces_the_window(self, qtbot, ds):
        v = _viewer(qtbot)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(_bold(ds), ds)
        first = v.show_header()
        second = v.show_header()
        qtbot.addWidget(second)
        assert first is not second
        assert not first.isVisible()


class TestFromAFinding:
    def test_a_header_finding_offers_the_button(self, qtbot):
        from PyQt6.QtWidgets import QPushButton

        from bidsmgr.gui.widgets.val_message import ValMessage

        msg = ValMessage("err", "REPETITION_TIME_MISMATCH", "differs",
                         view_label="Show in the header")
        qtbot.addWidget(msg)
        buttons = [b for b in msg.findChildren(QPushButton) if b.text() == "Show in the header"]
        assert buttons
        with qtbot.waitSignal(msg.view_requested, timeout=1000) as got:
            buttons[0].click()
        assert got.args == ["REPETITION_TIME_MISMATCH"]

    def test_the_editor_opens_the_image_on_the_row(self, qtbot, ds):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(ds, persist=False)
        ep._on_header_requested(_bold(ds), "REPETITION_TIME_MISMATCH")
        viewer = ep._nifti_viewer
        qtbot.waitUntil(lambda: viewer.presenter.header_dialog is not None, timeout=20_000)
        dlg = viewer.presenter.header_dialog
        qtbot.addWidget(dlg)
        assert viewer.current_file() == _bold(ds)
        assert dlg.tree.currentItem().text(0).strip() == "Repetition time"

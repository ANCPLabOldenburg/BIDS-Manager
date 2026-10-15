"""The Editor's Dashboard opens and shows its headline numbers.

Its number tiles are ``dialog_chrome.stat``, shared with the Converter's
Overview; this is the Dashboard's own check that they still read right.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QLabel

pytestmark = pytest.mark.gui


def _dataset(root: Path) -> Path:
    for sub in ("sub-01", "sub-02"):
        anat = root / sub / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_T1w.nii.gz").write_bytes(b"")
        (anat / f"{sub}_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text('{"Name": "Two", "BIDSVersion": "1.10.0"}')
    (root / "participants.tsv").write_text("participant_id\nsub-01\nsub-02\n")
    return root


def test_the_dashboard_counts_the_dataset(qtbot, tmp_path: Path) -> None:
    from bidsmgr.gui.dashboard_dialog import DashboardDialog

    dialog = DashboardDialog(_dataset(tmp_path / "two"))
    qtbot.addWidget(dialog)
    numbers = [label.text() for label in dialog.findChildren(QLabel, "stat-value-text")]
    captions = [label.text() for label in dialog.findChildren(QLabel, "stat-caption")]
    assert captions[:4] == ["subjects", "sessions", "datatypes", "suffixes"]
    assert numbers[:4] == ["2", "0", "1", "1"]


def test_a_stat_tile_can_be_updated_in_place(qtbot) -> None:
    from bidsmgr.gui.dialog_chrome import stat

    tile = stat("subjects", "0")
    qtbot.addWidget(tile)
    tile.number.setText("12")
    assert tile.number.objectName() == "stat-value-text"
    assert tile.caption.text() == "subjects"
    assert tile.findChild(QLabel, "stat-value-text").text() == "12"

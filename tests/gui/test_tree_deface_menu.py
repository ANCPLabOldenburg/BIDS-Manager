"""The two defacing entries on the tree's right-click.

Neither had a test. The menu is where both features are actually reached, so
an entry that silently stops being offered is a feature that silently stops
existing, and the whole-multi-selection behaviour is the part a user would
notice only by defacing the wrong files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QMenu, QTreeWidgetItemIterator  # noqa: E402

from .conftest import open_every_folder
from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE, BidsTreePane  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 64)
    (anat / "sub-01_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


@pytest.fixture
def pane(qtbot, dataset: Path) -> BidsTreePane:
    widget = BidsTreePane()
    qtbot.addWidget(widget)
    widget.set_root(dataset)
    return widget


def _item_for(pane: BidsTreePane, name: str):
    """Find the tree row whose path ends in *name*, expanding as needed."""
    open_every_folder(pane._tree)
    it = QTreeWidgetItemIterator(pane._tree)
    while it.value():
        item = it.value()
        value = item.data(0, PATH_ROLE)
        if value and Path(value).name == name:
            return item
        it += 1
    raise AssertionError(f"no row for {name}")


def _menu_labels(pane: BidsTreePane, monkeypatch, name: str) -> list[str]:
    """Open the context menu on *name* and return what it offered."""
    item = _item_for(pane, name)
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    captured: list[str] = []

    def _exec(self, *args, **kwargs):
        captured.extend(a.text() for a in self.actions() if a.text())
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))
    return captured


def test_an_image_offers_both_removing_the_face_and_checking_it(
    pane: BidsTreePane, monkeypatch,
) -> None:
    labels = _menu_labels(pane, monkeypatch, "sub-01_T1w.nii.gz")
    assert "Remove faces..." in labels
    assert "Compare with the original..." in labels


def test_a_sidecar_offers_removing_but_not_comparing(
    pane: BidsTreePane, monkeypatch,
) -> None:
    """Comparing is per IMAGE. There is nothing to show for a .json."""
    labels = _menu_labels(pane, monkeypatch, "sub-01_T1w.json")
    assert "Remove faces..." in labels
    assert "Compare with the original..." not in labels


def test_a_folder_offers_removing_but_not_comparing(
    pane: BidsTreePane, monkeypatch,
) -> None:
    labels = _menu_labels(pane, monkeypatch, "anat")
    assert "Remove faces..." in labels
    assert "Compare with the original..." not in labels


def test_removing_faces_emits_the_clicked_scope(
    pane: BidsTreePane, monkeypatch, dataset: Path,
) -> None:
    item = _item_for(pane, "sub-01_T1w.nii.gz")
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    seen: list[list] = []
    pane.deface_requested.connect(seen.append)

    def _exec(self, *args, **kwargs):
        for action in self.actions():
            if action.text() == "Remove faces...":
                action.trigger()
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))

    assert seen == [[dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"]]


def test_comparing_emits_the_file_itself_not_a_list(
    pane: BidsTreePane, monkeypatch, dataset: Path,
) -> None:
    """One image, so one Path. A list would make the handler guess."""
    item = _item_for(pane, "sub-01_T1w.nii.gz")
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    seen: list = []
    pane.deface_compare_requested.connect(seen.append)

    def _exec(self, *args, **kwargs):
        for action in self.actions():
            if action.text() == "Compare with the original...":
                action.trigger()
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))

    assert seen == [dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"]


def test_the_tree_offers_skull_stripping_too(pane: BidsTreePane, monkeypatch):
    labels = _menu_labels(pane, monkeypatch, "sub-01_T1w.nii.gz")
    assert "Remove the skull..." in labels
    assert "Put the face back..." in labels


def test_stripping_emits_the_clicked_scope(
    pane: BidsTreePane, monkeypatch, dataset: Path,
) -> None:
    item = _item_for(pane, "sub-01_T1w.nii.gz")
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    seen: list[list] = []
    pane.strip_requested.connect(seen.append)

    def _exec(self, *args, **kwargs):
        for action in self.actions():
            if action.text() == "Remove the skull...":
                action.trigger()
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))

    assert seen == [[dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"]]


# ---------------------------------------------------------------------------
# Comparing any two images, not just a defacing pair.


def test_any_nifti_offers_a_comparison(pane: BidsTreePane, monkeypatch) -> None:
    labels = _menu_labels(pane, monkeypatch, "sub-01_T1w.nii.gz")
    assert "Compare with another image..." in labels


def test_a_sidecar_does_not_offer_a_comparison(
    pane: BidsTreePane, monkeypatch,
) -> None:
    """Comparing is per IMAGE. There is nothing to show for a .json."""
    labels = _menu_labels(pane, monkeypatch, "sub-01_T1w.json")
    assert not [label for label in labels if label.startswith("Compare with")]


def test_comparing_emits_the_clicked_image(
    pane: BidsTreePane, monkeypatch, dataset: Path,
) -> None:
    item = _item_for(pane, "sub-01_T1w.nii.gz")
    monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
    seen: list[list] = []
    pane.compare_requested.connect(seen.append)

    def _exec(self, *args, **kwargs):
        for action in self.actions():
            if action.text().startswith("Compare with"):
                action.trigger()
        return None

    monkeypatch.setattr(QMenu, "exec", _exec)
    pane._on_show_context_menu(QPoint(1, 1))

    assert seen == [[dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"]]

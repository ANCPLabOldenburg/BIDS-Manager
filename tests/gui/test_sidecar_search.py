"""Searching a sidecar, as the DICOM header is searched.

One field, shared with the DICOM pane, finds a field by its name, its value
or what the standard says it is, in both views. In Tree view a key nested in
a field is found and shown with the field that holds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from bidsmgr.gui.widgets.search_field import SearchField
from bidsmgr.gui.widgets.sidecar_form_pane import SidecarFormPane

pytestmark = pytest.mark.gui


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_T1w.json").write_text(json.dumps({
        "Manufacturer": "Siemens",
        "EchoTime": 0.003,
        "RepetitionTime": 2.3,
        "ProcedureStepDescription": "Head^Research",
        "DeidentificationMethodCodeSequence": [
            {"CodeValue": "113100", "CodeMeaning": "Basic Application Profile"},
        ],
    }))
    (anat / "sub-01_T2w.nii.gz").write_bytes(b"")
    (anat / "sub-01_T2w.json").write_text(json.dumps({
        "Manufacturer": "Philips", "EchoTime": 0.1}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Study", "BIDSVersion": "1.10.0"}))
    return root


@pytest.fixture
def pane(qtbot, root: Path) -> SidecarFormPane:
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.resize(560, 700)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_file(root / "sub-01" / "anat" / "sub-01_T1w.json", root, None)
    return pane


def _shown(pane: SidecarFormPane) -> set[str]:
    return {row._key for row in pane._rows if not row.isHidden()}


def test_the_viewers_share_one_search_field(pane) -> None:
    from bidsmgr.gui.widgets.dicom_pane import DicomPane

    assert isinstance(pane.search, SearchField)
    assert isinstance(DicomPane().search, SearchField)
    assert pane.search.isClearButtonEnabled()
    assert pane.search.actions(), "a magnifier says what the box is for"


def test_a_field_is_found_by_its_name(pane) -> None:
    every = _shown(pane)
    pane.search.setText("echo")
    shown = _shown(pane)
    assert "EchoTime" in shown and "Manufacturer" not in shown
    assert shown < every
    assert pane._match_count.isVisible()
    assert pane._match_count.text() == f"{len(shown)} of {len(pane._rows)}"
    pane.search.clear()
    assert _shown(pane) == every and not pane._match_count.isVisible()


def test_a_field_is_found_by_its_value_and_by_what_it_means(pane) -> None:
    pane.search.setText("SIEMENS")          # case does not matter
    shown = _shown(pane)
    assert "Manufacturer" in shown, "found by its value"
    # Every row shown says it somewhere: the standard's description of
    # DwellTime, for one, is about Siemens scanners.
    rows = {row._key: row for row in pane._rows}
    assert all("siemens" in pane._row_text(rows[k]) for k in shown)
    assert len(shown) < len(rows)
    # Found by the schema's description of it, not by its name or value:
    # the standard says MagneticFieldStrength is in Tesla.
    pane.search.setText("tesla")
    assert "MagneticFieldStrength" in _shown(pane)


def test_a_value_typed_a_moment_ago_is_found(pane) -> None:
    pane._json_cache["Manufacturer"] = "GE HealthCare"
    pane.search.setText("healthcare")
    assert _shown(pane) == {"Manufacturer"}


def test_nothing_found_says_so(pane) -> None:
    pane.search.setText("no such field anywhere")
    assert _shown(pane) == set()
    assert pane._no_match.isVisible()
    assert "no such field anywhere" in pane._no_match.text()
    assert pane._match_count.text().startswith("0 of")


def test_the_search_follows_the_user_to_the_next_file(pane, root: Path) -> None:
    pane.search.setText("philips")
    pane.set_file(root / "sub-01" / "anat" / "sub-01_T2w.json", root, None)
    assert pane.search.text() == "philips"
    assert "Manufacturer" in _shown(pane), "the next file is searched as it opens"
    assert len(_shown(pane)) < len(pane._rows)


def test_tree_view_finds_a_key_inside_a_field(pane) -> None:
    pane._tree_view_btn.setChecked(True)
    pane._view_group.idClicked.emit(1)
    tree = pane._tree_view
    pane.search.setText("codemeaning")
    visible = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if not tree.topLevelItem(i).isHidden()]
    assert [i.text(0) for i in visible] == ["DeidentificationMethodCodeSequence"]
    holder = visible[0]
    assert holder.isExpanded(), "opened, so the match can be seen"
    # A match shows everything inside it.
    pane.search.setText("DeidentificationMethodCodeSequence")
    entry = holder.child(0)
    assert not entry.isHidden()
    assert all(not entry.child(j).isHidden() for j in range(entry.childCount()))
    pane.search.clear()
    assert all(not tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount()))


def test_the_find_key_reaches_the_field_and_escape_clears_it(qtbot, pane) -> None:
    pane._rows[0].setFocus()
    QTest.keyClick(pane._rows[0], Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(pane.search.hasFocus, timeout=2000)
    QTest.keyClicks(pane.search, "echo")
    assert pane.search.text() == "echo"
    QTest.keyClick(pane.search, Qt.Key.Key_Escape)
    assert pane.search.text() == ""


def test_both_views_find_the_same_fields_before_any_validation(pane) -> None:
    """Tree view searches what the standard says a field is, like the BIDS
    view, without waiting for a validation report to know the file's kind."""
    pane.search.setText("tesla")
    in_form = _shown(pane)
    pane._tree_view_btn.setChecked(True)
    pane._view_group.idClicked.emit(1)
    tree = pane._tree_view
    in_tree = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
               if not tree.topLevelItem(i).isHidden()}
    assert "MagneticFieldStrength" in in_tree
    assert in_tree == in_form

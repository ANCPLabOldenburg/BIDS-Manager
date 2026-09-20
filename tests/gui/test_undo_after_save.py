"""Undo has to undo, including after a save.

The report: "undo and redo are not working completely fine in the editor tab,
specially when we save the changes in a sidecar."

Reproduced exactly. Undo restored the in-memory cache and nothing else, so
after a save the FILE kept the written value, the pane went dirty again, and
the button looked like it did nothing. Half an undo is worse than none,
because it leaves the screen and the disk disagreeing.

The rule now is the one a user already holds: what you see is what is on disk,
unless you have unsaved changes. So an undo from a CLEAN state writes the file
back, and an undo from a dirty one does not, because in that case nothing had
reached the disk to revert.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.gui.widgets.sidecar_form_pane import SCOPE_PRESENT, SidecarFormPane
from bidsmgr.gui.widgets.sidecar_row import SidecarRow

pytestmark = pytest.mark.gui


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    path = folder / "sub-01_T1w.json"
    path.write_text(json.dumps({"EchoTime": 0.03, "FlipAngle": 9}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return path


@pytest.fixture
def pane(qtbot, sidecar: Path) -> SidecarFormPane:
    widget = SidecarFormPane()
    qtbot.addWidget(widget)
    widget.set_autosave(False)
    widget.set_field_scope(SCOPE_PRESENT)
    widget.set_file(sidecar, sidecar.parents[2], None)
    return widget


def _edit(pane: SidecarFormPane, key: str, text: str) -> None:
    row = next(r for r in pane.findChildren(SidecarRow) if r.key == key)
    row.editor().setText(text)
    row.editor().editingFinished.emit()


def _disk(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# The reported case
# ---------------------------------------------------------------------------


def test_undo_after_a_save_puts_the_file_back(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    _edit(pane, "EchoTime", "0.09")
    assert pane.save()
    assert _disk(sidecar)["EchoTime"] == 0.09

    pane.undo()
    assert _disk(sidecar)["EchoTime"] == 0.03, "the FILE must go back"
    assert not pane.is_dirty(), "and it is not left looking unsaved"


def test_redo_after_that_puts_it_forward_again(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    _edit(pane, "EchoTime", "0.09")
    pane.save()
    pane.undo()
    pane.redo()
    assert _disk(sidecar)["EchoTime"] == 0.09
    assert not pane.is_dirty()


def test_several_saved_edits_unwind_one_at_a_time(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    for value in ("0.04", "0.05", "0.06"):
        _edit(pane, "EchoTime", value)
        pane.save()
    for expected in (0.05, 0.04, 0.03):
        pane.undo()
        assert _disk(sidecar)["EchoTime"] == expected


# ---------------------------------------------------------------------------
# And what must NOT change
# ---------------------------------------------------------------------------


def test_an_unsaved_edit_is_undone_in_memory_only(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    """Nothing reached the disk, so there is nothing there to revert, and
    writing would turn an undo into a save."""
    before = _disk(sidecar)
    _edit(pane, "EchoTime", "0.09")
    assert _disk(sidecar) == before, "the edit was never written"
    pane.undo()
    assert _disk(sidecar) == before, "and undoing it writes nothing either"
    assert not pane.is_dirty()


def test_undoing_past_a_save_into_unsaved_territory(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    """Save, then edit again without saving, then undo. The pane returns to
    the saved state, and the file already holds it, so nothing is written."""
    _edit(pane, "EchoTime", "0.09")
    pane.save()
    _edit(pane, "EchoTime", "0.5")
    assert pane.is_dirty()

    pane.undo()
    assert pane._json_cache["EchoTime"] == 0.09
    assert _disk(sidecar)["EchoTime"] == 0.09
    assert not pane.is_dirty()


def test_undo_does_not_disturb_the_other_fields(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    _edit(pane, "EchoTime", "0.09")
    pane.save()
    pane.undo()
    assert _disk(sidecar)["FlipAngle"] == 9


def test_undo_with_nothing_to_undo_is_a_no_op(
    pane: SidecarFormPane, sidecar: Path,
) -> None:
    before = _disk(sidecar)
    pane.undo()
    pane.redo()
    assert _disk(sidecar) == before


def test_saving_keeps_the_history(pane: SidecarFormPane) -> None:
    """A save is not a new beginning. Clearing the stack on save is the other
    way this feature can look broken."""
    _edit(pane, "EchoTime", "0.09")
    pane.save()
    assert pane._history.can_undo


def test_reverting_clears_the_history(pane: SidecarFormPane) -> None:
    """Revert discards the edits, so their undo steps are meaningless."""
    _edit(pane, "EchoTime", "0.09")
    pane.revert()
    assert not pane._history.can_undo

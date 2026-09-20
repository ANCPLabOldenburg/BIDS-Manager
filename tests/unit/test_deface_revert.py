"""Putting the face back, from the copy that still has one.

Undo reverses the LAST operation, which is right five minutes after defacing
and useless a week later with a dozen edits on top. The ``sourcedata/`` mirror
outlives all of that, so restoring from it is the long-lived counterpart.

The two things worth pinning are the ones a user cannot check by eye: that the
sidecar stops CLAIMING the file is defaced (a restored image that still says it
was defaced will never be defaced again by anyone reading the metadata), and
that the mirror survives the restore, because deleting somebody's only
undefaced copy as a side effect has no way back.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from bidsmgr.deface import status
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE
from bidsmgr.deface.revert import revert_dataset, revertable
from bidsmgr.project.operations import begin_operation, read_log, undo_last
from bidsmgr.util.cancel import OperationCancelled

REL = "sub-01/anat/sub-01_T1w.nii.gz"
OTHER = "sub-01/anat/sub-01_T2w.nii.gz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    return root


def _defaced_with_mirror(root: Path, rel: str = REL) -> str:
    """An image that has been defaced, with the pristine copy in sourcedata/."""
    image = root / rel
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, image)
    pristine = _sha(image)

    mirror = root / "sourcedata" / rel
    mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, mirror)

    image.write_bytes(b"defaced bytes")
    sidecar = status.sidecar_for(image)
    sidecar.write_text(json.dumps(
        status.record({"Manufacturer": "Siemens"}, ALLINEATE)
    ))
    return pristine


def test_it_offers_only_what_we_defaced(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    # A mirror for something we never defaced must not be offered: restoring
    # it would be acting on somebody else's data on a guess.
    other = root / OTHER
    shutil.copyfile(TEMPLATE, other)
    mirror = root / "sourcedata" / OTHER
    mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, mirror)

    assert [r.relative for r in revertable(root)] == [REL]


def test_it_restores_the_bytes_exactly(tmp_path):
    root = _dataset(tmp_path)
    pristine = _defaced_with_mirror(root)

    out = revert_dataset(root)
    assert out.ok, out.failed
    assert out.reverted == [REL]
    assert _sha(root / REL) == pristine


def test_the_sidecar_stops_claiming_it_is_defaced(tmp_path):
    """Or nobody will ever deface it again; the metadata says it is done."""
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)

    revert_dataset(root)

    doc = json.loads(status.sidecar_for(root / REL).read_text())
    assert status.defaced_by_us(doc) is None
    assert not status.defaced_by_us_at_all(doc)
    assert doc["Manufacturer"] == "Siemens", "an unrelated field was lost"


def test_the_mirror_is_kept(tmp_path):
    """Reverting must not destroy the only undefaced copy that exists."""
    root = _dataset(tmp_path)
    pristine = _defaced_with_mirror(root)

    revert_dataset(root)

    mirror = root / "sourcedata" / REL
    assert mirror.is_file() and _sha(mirror) == pristine


def test_it_is_one_undoable_entry(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    _defaced_with_mirror(root, OTHER)

    out = revert_dataset(root)
    assert len(out.reverted) == 2
    log = read_log(root)
    assert len(log) == 1
    assert "2 images" in log[0]["label"]


def test_reverting_can_itself_be_undone(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    defaced = _sha(root / REL)

    revert_dataset(root)
    assert _sha(root / REL) != defaced

    undo_last(root)
    assert _sha(root / REL) == defaced
    doc = json.loads(status.sidecar_for(root / REL).read_text())
    assert status.defaced_by_us(doc) is not None


def test_only_narrows_it_to_a_part_ticked_preview(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    _defaced_with_mirror(root, OTHER)
    kept = _sha(root / OTHER)

    out = revert_dataset(root, only=[REL])

    assert out.reverted == [REL]
    assert _sha(root / OTHER) == kept, "an unticked image was restored anyway"


def test_it_also_works_from_the_edit_history(tmp_path):
    """No sourcedata/ mirror, but the operations log kept the original."""
    root = _dataset(tmp_path)
    image = root / REL
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, image)
    pristine = _sha(image)
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.write_bytes(image, b"defaced")
        op.write_json(
            status.sidecar_for(image), status.record({}, ALLINEATE),
        )

    assert [r.source for r in revertable(root)] == ["history"]
    assert revert_dataset(root).ok
    assert _sha(image) == pristine


def test_nothing_to_revert_is_not_an_error(tmp_path):
    root = _dataset(tmp_path)
    shutil.copyfile(TEMPLATE, (root / REL))
    out = revert_dataset(root)
    assert out.reverted == [] and out.failed == [] and not out.cancelled


def test_stopping_rolls_the_whole_thing_back(tmp_path):
    """A half-reverted dataset is as bad as a half-defaced one."""
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    _defaced_with_mirror(root, OTHER)
    before = {rel: _sha(root / rel) for rel in (REL, OTHER)}

    calls = {"n": 0}

    def _cancel_after_one():
        calls["n"] += 1
        if calls["n"] > 1:
            raise OperationCancelled("stop")

    out = revert_dataset(root, cancel_check=_cancel_after_one)

    assert out.cancelled and not out.ok
    assert out.reverted == []
    for rel, sha in before.items():
        assert _sha(root / rel) == sha, f"{rel} was left restored"


def test_a_target_narrows_the_walk(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    (root / "sub-02" / "anat").mkdir(parents=True)
    _defaced_with_mirror(root, "sub-02/anat/sub-02_T1w.nii.gz")

    found = revertable(root, [root / "sub-02"])
    assert [r.relative for r in found] == ["sub-02/anat/sub-02_T1w.nii.gz"]


def test_an_original_that_vanished_is_reported_not_fatal(tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    items = revertable(root)
    (root / "sourcedata" / REL).unlink()

    out = revert_dataset(root, items=items)

    assert out.reverted == []
    assert out.skipped and out.skipped[0][0] == REL
    assert not out.failed


@pytest.mark.parametrize("name", ["revert_dataset", "revertable"])
def test_the_module_exports_it(name):
    import bidsmgr.deface.revert as mod

    assert hasattr(mod, name)

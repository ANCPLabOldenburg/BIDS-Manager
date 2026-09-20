"""Finding the undefaced copy, and saying why there isn't one.

The lookup itself is small. What is worth pinning is the order it prefers, and
the fact that the explanation for a missing original is never blank: "defaced
during conversion, so one never existed" is a correct answer, and a user who
sees an empty pane instead will assume the tool is broken.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bidsmgr.deface import compare, status
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE
from bidsmgr.project.operations import begin_operation

REL = "sub-01/anat/sub-01_T1w.nii.gz"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    shutil.copyfile(TEMPLATE, root / REL)
    return root


def _mark_defaced(root: Path) -> None:
    sidecar = status.sidecar_for(root / REL)
    sidecar.write_text(json.dumps(status.record({}, ALLINEATE)))


def test_nothing_kept_means_no_original(tmp_path):
    root = _dataset(tmp_path)
    assert compare.original_for(root, REL) is None


def test_the_sourcedata_mirror_is_found(tmp_path):
    root = _dataset(tmp_path)
    mirror = root / "sourcedata" / REL
    mirror.parent.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, mirror)

    found = compare.original_for(root, REL)
    assert found is not None
    assert found.path == mirror
    assert found.source == "sourcedata"
    assert "sourcedata" in found.description


def test_the_edit_history_is_found(tmp_path):
    root = _dataset(tmp_path)
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.write_bytes(root / REL, b"defaced")

    found = compare.original_for(root, REL)
    assert found is not None
    assert found.source == "history"
    assert found.path.read_bytes() == TEMPLATE.read_bytes()
    assert "Deface 1 image" in found.description


def test_the_earliest_operation_wins_not_the_latest(tmp_path):
    """A second defacing saves the already-defaced image, not the pristine one."""
    root = _dataset(tmp_path)
    pristine = (root / REL).read_bytes()
    with begin_operation(root, "first") as op:
        op.write_bytes(root / REL, b"once")
    with begin_operation(root, "second") as op:
        op.write_bytes(root / REL, b"twice")

    found = compare.original_for(root, REL)
    assert found is not None
    assert found.path.read_bytes() == pristine, (
        "picked the intermediate version, so the comparison would be useless"
    )
    assert found.label == "first"


def test_sourcedata_wins_over_the_history(tmp_path):
    """It is never overwritten, so it is pristine however many runs happened."""
    root = _dataset(tmp_path)
    mirror = root / "sourcedata" / REL
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"pristine")
    with begin_operation(root, "later") as op:
        op.write_bytes(root / REL, b"defaced")

    found = compare.original_for(root, REL)
    assert found is not None and found.source == "sourcedata"


def test_a_missing_file_is_named_as_such(tmp_path):
    root = _dataset(tmp_path)
    assert "not in this dataset" in compare.explain_missing(root, "sub-09/x.nii.gz")


def test_an_undefaced_image_is_told_to_deface_first(tmp_path):
    root = _dataset(tmp_path)
    msg = compare.explain_missing(root, REL)
    assert "has not been defaced" in msg
    assert "Tools" in msg


def test_a_convert_time_deface_is_explained_as_deliberate(tmp_path):
    """The one case where a missing original is the feature working."""
    root = _dataset(tmp_path)
    _mark_defaced(root)

    msg = compare.explain_missing(root, REL)
    assert "during conversion" in msg
    assert "deliberate" in msg
    assert "sourcedata" in msg, "does not say how to keep one next time"


def test_another_tool_s_defacing_is_distinguished(tmp_path):
    root = _dataset(tmp_path)
    status.sidecar_for(root / REL).write_text(json.dumps({
        status.METHOD: ["pydeface 2.0"],
    }))

    msg = compare.explain_missing(root, REL)
    assert "another tool" in msg


def test_comparable_lists_only_images_with_both_halves(tmp_path):
    root = _dataset(tmp_path)
    # An original with no image left in the dataset must not be offered.
    with begin_operation(root, "deface") as op:
        op.write_bytes(root / REL, b"defaced")
    gone = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    shutil.copyfile(TEMPLATE, gone)
    with begin_operation(root, "delete") as op:
        op.delete(gone)

    listed = compare.comparable(root)
    assert REL in listed
    assert "sub-01/anat/sub-01_T2w.nii.gz" not in listed


def test_relative_paths_are_posix_on_every_platform(tmp_path):
    root = _dataset(tmp_path)
    mirror = root / "sourcedata" / REL
    mirror.parent.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, mirror)
    assert all("\\" not in rel for rel in compare.comparable(root))


def test_a_windows_style_relative_path_still_resolves(tmp_path):
    """Callers hand us what the tree gave them; do not fail on a backslash."""
    root = _dataset(tmp_path)
    mirror = root / "sourcedata" / REL
    mirror.parent.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, mirror)
    assert compare.original_for(root, REL.replace("/", "\\")) is not None


@pytest.mark.parametrize("name", ["Original", "original_for", "explain_missing"])
def test_the_package_exports_it(name):
    import bidsmgr.deface as pkg

    assert hasattr(pkg, name)

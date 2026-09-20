"""Binary writes are as reversible as text ones.

Defacing replaces a ``.nii.gz`` with new image bytes. Until these existed the
only reversible writes were ``write_text`` and ``write_json``, so the one
operation that has to be undoable, because it is destructive and irreversible
by hand, had no primitive it could use.

The round trip is the test that matters: deface, undo, and the file is
byte-identical to what was there. Everything else here supports that.
"""

from __future__ import annotations

import os

import pytest

from bidsmgr.project.operations import (
    OperationError,
    begin_operation,
    read_log,
    undo_last,
)

# Bytes that a text round trip would mangle: a NUL, a lone high byte that is
# not valid UTF-8, and a CR LF pair that newline translation would rewrite.
BINARY = b"\x1f\x8b\x08\x00nifti\x00\xff\xfe\r\n\x00tail"
OTHER = b"\x1f\x8b\x08\x00defaced\x00\x01\x02\x03"


def _dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text("{}")
    return root


def test_write_bytes_then_undo_restores_the_original(tmp_path):
    root = _dataset(tmp_path)
    target = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    target.write_bytes(BINARY)

    with begin_operation(root, "Deface 1 file") as op:
        op.write_bytes(target, OTHER)

    assert target.read_bytes() == OTHER
    undo_last(root)
    assert target.read_bytes() == BINARY


def test_replace_from_streams_another_file_into_place(tmp_path):
    root = _dataset(tmp_path)
    target = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    target.write_bytes(BINARY)
    produced = tmp_path / "engine-output.nii.gz"
    produced.write_bytes(OTHER)

    with begin_operation(root, "Deface 1 file") as op:
        op.replace_from(target, produced)

    assert target.read_bytes() == OTHER
    # The source is the caller's, and it survives: a failed replace must not
    # destroy the only copy of the new bytes.
    assert produced.read_bytes() == OTHER

    undo_last(root)
    assert target.read_bytes() == BINARY


def test_replace_from_refuses_a_missing_source(tmp_path):
    root = _dataset(tmp_path)
    target = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    target.write_bytes(BINARY)

    with pytest.raises(OperationError):
        with begin_operation(root, "Deface") as op:
            op.replace_from(target, tmp_path / "not-there.nii.gz")

    assert target.read_bytes() == BINARY


def test_a_failure_part_way_rolls_the_whole_thing_back(tmp_path):
    """Two files defaced, the second fails: neither is left changed."""
    root = _dataset(tmp_path)
    first = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    second = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    first.write_bytes(BINARY)
    second.write_bytes(BINARY)

    with pytest.raises(OperationError):
        with begin_operation(root, "Deface 2 files") as op:
            op.replace_from(first, _produced(tmp_path, OTHER))
            op.replace_from(second, tmp_path / "gone.nii.gz")

    assert first.read_bytes() == BINARY
    assert second.read_bytes() == BINARY


def test_the_temp_file_lands_beside_the_target(tmp_path, monkeypatch):
    """So the final move is a rename within one filesystem.

    ``os.replace`` fails with EXDEV across devices, which is exactly what a
    temp directory can be. This asserts the staging path rather than the
    outcome, because the outcome is identical either way until the day
    somebody runs it with /tmp on a different mount.
    """
    root = _dataset(tmp_path)
    target = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    target.write_bytes(BINARY)
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst, *a, **kw):
        seen.append(str(src))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", spy)
    with begin_operation(root, "Deface") as op:
        op.replace_from(target, _produced(tmp_path, OTHER))

    assert seen, "nothing was moved into place"
    assert all(
        os.path.dirname(p) == str(target.parent) for p in seen
    ), f"staged outside the target directory: {seen}"


def test_the_operation_is_one_entry_in_the_history(tmp_path):
    root = _dataset(tmp_path)
    a = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    b = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    a.write_bytes(BINARY)
    b.write_bytes(BINARY)

    with begin_operation(root, "Deface 2 files") as op:
        op.replace_from(a, _produced(tmp_path, OTHER, "a"))
        op.replace_from(b, _produced(tmp_path, OTHER, "b"))

    log = read_log(root)
    assert len(log) == 1
    assert log[0]["label"] == "Deface 2 files"

    undo_last(root)
    assert a.read_bytes() == BINARY
    assert b.read_bytes() == BINARY


def _produced(tmp_path, data: bytes, name: str = "out") -> object:
    p = tmp_path / f"{name}.nii.gz"
    p.write_bytes(data)
    return p

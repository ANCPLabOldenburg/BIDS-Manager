"""Tests for the scan-version engine (``bidsmgr.project.workspace``).

Each scan of a source folder gets its own version dir under
``.bidsmgr/project/scans/``, so a second scan never overwrites the first.
"""

from __future__ import annotations

from pathlib import Path

from bidsmgr.project import workspace as ws


def _make_version(root: Path, label: str, raw: str) -> Path:
    vdir = ws.allocate_version_dir(root, label)
    vdir.mkdir(parents=True)
    ws.version_inventory(vdir).write_text("participant_id\tinclude\n")
    ws.write_version_meta(vdir, source_label=label, raw_root=raw, status="curating")
    return vdir


def test_allocate_increments_index(tmp_path: Path):
    root = tmp_path / "ds"
    v1 = _make_version(root, "first", "/data/a")
    v2 = ws.allocate_version_dir(root, "second")
    assert v1.name == "0001__first"
    assert v2.name == "0002__second"


def test_allocate_makes_a_filesystem_safe_slug(tmp_path: Path):
    root = tmp_path / "ds"
    v = ws.allocate_version_dir(root, "weird/name:with*chars")
    # No path separators / illegal chars leak into the version folder name.
    assert "/" not in v.name and ":" not in v.name and "*" not in v.name
    assert v.name.startswith("0001__")


def test_list_versions_sorted_and_typed(tmp_path: Path):
    root = tmp_path / "ds"
    _make_version(root, "alpha", "/data/a")
    _make_version(root, "beta", "/data/b")
    versions = ws.list_versions(root)
    assert [v.index for v in versions] == [1, 2]
    assert versions[0].source_label == "alpha"
    assert versions[1].raw_root == "/data/b"
    assert all(v.status == "curating" for v in versions)


def test_incomplete_version_without_inventory_is_skipped(tmp_path: Path):
    root = tmp_path / "ds"
    _make_version(root, "good", "/data/a")
    # An allocated-but-never-finished scan dir (no inventory) is ignored.
    bad = ws.allocate_version_dir(root, "aborted")
    bad.mkdir(parents=True)
    versions = ws.list_versions(root)
    assert [v.version_id for v in versions] == ["0001__good"]


def test_write_version_meta_merges(tmp_path: Path):
    root = tmp_path / "ds"
    vdir = _make_version(root, "lab", "/data/a")
    ws.write_version_meta(vdir, source_label="lab", raw_root="/data/a", status="converted")
    meta = ws.read_version_meta(vdir)
    assert meta["status"] == "converted"
    assert meta["source_label"] == "lab"


def test_list_versions_empty_when_no_scans(tmp_path: Path):
    assert ws.list_versions(tmp_path / "nope") == []

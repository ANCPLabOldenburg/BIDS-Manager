"""Moving through the dataset: files that differ in one entity, fieldmaps
meant for an image."""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.viz.bids import fieldmaps_for, neighbours, siblings_along


def _touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _ds(root: Path) -> Path:
    (root / "dataset_description.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text("{}")
    for sub in ("01", "02", "10"):
        for ses in ("1", "2"):
            for run in ("1", "2", "10"):
                _touch(root, f"sub-{sub}/ses-{ses}/func/"
                             f"sub-{sub}_ses-{ses}_task-x_run-{run}_bold.nii.gz")
    # Different in more than one entity, or another suffix or extension.
    _touch(root, "sub-01/ses-1/func/sub-01_ses-1_task-y_run-2_bold.nii.gz")
    _touch(root, "sub-01/ses-1/func/sub-01_ses-1_task-x_run-2_sbref.nii.gz")
    _touch(root, "sub-01/ses-1/func/sub-01_ses-1_task-x_run-2_bold.json")
    return root


def _p(root, sub="01", ses="1", run="1"):
    return root / f"sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-x_run-{run}_bold.nii.gz"


def test_runs_sort_as_numbers(tmp_path):
    root = _ds(tmp_path)
    runs = [p.name.split("_run-")[1].split("_")[0] for p in siblings_along(_p(root), "run")]
    assert runs == ["1", "2", "10"]


def test_only_files_differing_in_that_entity_count(tmp_path):
    root = _ds(tmp_path)
    found = siblings_along(_p(root), "run")
    assert all("task-x" in p.name and p.name.endswith("_bold.nii.gz") for p in found)
    assert all(p.parent == _p(root).parent for p in found)


def test_subjects_and_sessions_are_found_through_their_folders(tmp_path):
    root = _ds(tmp_path)
    subs = siblings_along(_p(root, run="2"), "sub", root)
    assert [p.parts[-4] for p in subs] == ["sub-01", "sub-02", "sub-10"]
    sessions = siblings_along(_p(root, sub="02"), "ses", root)
    assert [p.parts[-3] for p in sessions] == ["ses-1", "ses-2"]


def test_neighbours_at_the_ends(tmp_path):
    root = _ds(tmp_path)
    nb = neighbours(_p(root, run="1"), root)
    assert nb["run"][0] is None and nb["run"][1] == _p(root, run="2")
    nb = neighbours(_p(root, sub="10", run="10"), root)
    assert nb["run"] == (_p(root, sub="10", run="2"), None)
    assert nb["sub"] == (_p(root, sub="02", run="10"), None)
    assert "echo" not in nb


def test_run_1_is_not_run_10s_neighbour_by_name(tmp_path):
    root = _ds(tmp_path)
    nb = neighbours(_p(root, run="2"), root)
    assert nb["run"] == (_p(root, run="1"), _p(root, run="10"))


def test_the_root_is_found_without_being_given(tmp_path):
    root = _ds(tmp_path / "Study")
    assert neighbours(_p(root, sub="02"))["sub"][0] == _p(root, sub="01")


class TestFieldmaps:
    def _fmap(self, root: Path, refs) -> Path:
        img = _touch(root, "sub-01/ses-1/fmap/sub-01_ses-1_phasediff.nii.gz")
        img.with_name("sub-01_ses-1_phasediff.json").write_text(
            json.dumps({"IntendedFor": refs}))
        return img

    def test_subject_relative_intended_for(self, tmp_path):
        root = _ds(tmp_path)
        fmap = self._fmap(root, ["ses-1/func/sub-01_ses-1_task-x_run-1_bold.nii.gz"])
        assert fieldmaps_for(_p(root), root) == [fmap]
        assert fieldmaps_for(_p(root, run="2"), root) == []

    def test_a_bids_uri(self, tmp_path):
        root = _ds(tmp_path)
        fmap = self._fmap(root, "bids::sub-01/ses-1/func/sub-01_ses-1_task-x_run-2_bold.nii.gz")
        assert fieldmaps_for(_p(root, run="2"), root) == [fmap]

    def test_another_subjects_fieldmap_is_not_this_ones(self, tmp_path):
        root = _ds(tmp_path)
        self._fmap(root, ["ses-1/func/sub-01_ses-1_task-x_run-1_bold.nii.gz"])
        assert fieldmaps_for(_p(root, sub="02"), root) == []

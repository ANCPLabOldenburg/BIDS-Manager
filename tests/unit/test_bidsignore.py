"""What .bidsignore hides, what the Editor offers to hide, and that the
Editor and the validator agree on both.

Reported 2026-10-19: picking one file and "Ignore selected files" ignored the
whole subject (the pane suggested the file's TOP folder), and the Editor
matched patterns with a translator of its own that read ``sub-01/anat`` as
matching at any depth, where the validator anchors it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import bidsignore as bi
from bidsmgr.project.operations import peek_undo, undo_last
from bidsmgr.vendor.bidsval.files.bidsignore import load_bidsignore

FILES = [
    "dataset_description.json",
    "participants.tsv",
    "sub-01/anat/sub-01_T1w.nii.gz",
    "sub-01/anat/sub-01_T1w.json",
    "sub-01/anat/notes.txt",
    "sub-01/ses-1/func/sub-01_ses-1_task-x_bold.nii.gz",
    "sub-01/ses-1/func/scanner.log",
    "sub-02/ses-2/func/scanner.log",
    "sub-02/anat/sub-02_T1w.nii.gz",
    "sub-02/anat/notes.txt",
    "sub-02/eeg/sub-02_task-x_eeg.mat",
    "sub-02/eeg/sub-02_task-x_eeg.xyz",
    "sub-01/anat/stray.xyz",
    "derivatives/tool/sub-01/anat/sub-01_desc-x_T1w.nii.gz",
    "extra/a.txt",
    "extra/b.txt",
    "extra/c.csv",
    "sourcedata/sub-01/1.dcm",
    "code/run.py",
]


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for rel in FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if rel.endswith(".json") else "x")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    (root / ".bidsmgr").mkdir()
    (root / ".bidsmgr" / "state.json").write_text("{}")
    return root


def _paths(root: Path) -> list[str]:
    return bi.list_dataset(root).paths


def test_picking_one_file_ignores_that_file_not_its_subject(ds: Path) -> None:
    paths = _paths(ds)
    first = bi.choices_for(["sub-01/anat/notes.txt"], paths)[0]
    assert first.label == "This file"
    assert first.patterns == ("/sub-01/anat/notes.txt",) and first.files == 1
    bi.add_patterns(ds, first.patterns)
    assert set(bi.ignored_by(ds, paths)) == {"sub-01/anat/notes.txt"}

    from bidsmgr.vendor.bidsval import validate

    validated = {Path(v.path).as_posix() for v in validate(ds, read_headers=False).files}
    assert "sub-01/anat/notes.txt" not in validated
    assert "sub-01/anat/sub-01_T1w.nii.gz" in validated, "the subject is still validated"


@pytest.mark.parametrize("lines", [
    ["sub-01/anat"],
    ["*.log"],
    ["notes.txt"],
    ["/extra/"],
    ["**/eeg/*.mat"],
    ["extra/[ab].txt"],
    ["extra/[!a].txt"],
    ["*.txt", "!/sub-02/anat/notes.txt"],
    ["/sub-*/anat/notes.txt"],
])
def test_the_editor_and_the_validator_agree(ds: Path, lines: list[str]) -> None:
    (ds / ".bidsignore").write_text("\n".join(lines) + "\n")
    paths = _paths(ds)
    validator = load_bidsignore(ds, include_defaults=False)
    assert set(bi.ignored_by(ds, paths)) == {p for p in paths if validator.match(p)}


def test_a_slash_in_the_middle_anchors_at_the_root(ds: Path) -> None:
    (ds / ".bidsignore").write_text("sub-01/anat\n")
    ignored = set(bi.ignored_by(ds, _paths(ds)))
    assert "sub-01/anat/notes.txt" in ignored
    assert "derivatives/tool/sub-01/anat/sub-01_desc-x_T1w.nii.gz" not in ignored


def test_a_character_class_is_a_class(ds: Path) -> None:
    assert bi.matches_for("extra/[ab].txt", _paths(ds)) == ["extra/a.txt", "extra/b.txt"]
    assert bi.matches_for("extra/[!a].txt", _paths(ds)) == ["extra/b.txt"]


def test_the_listing_leaves_out_what_no_pattern_can_concern(ds: Path) -> None:
    listing = bi.list_dataset(ds)
    assert sorted(listing.standard) == ["code", "sourcedata"]
    assert not any(p.startswith(("sourcedata/", "code/", ".bidsmgr/")) for p in listing.paths)
    assert "sub-01/anat/notes.txt" in listing.paths


def test_the_choices_widen_step_by_step(ds: Path) -> None:
    paths = _paths(ds)
    image = {c.key: c for c in bi.choices_for(["sub-01/anat/sub-01_T1w.nii.gz"], paths)}
    assert image["recording"].patterns == ("/sub-01/anat/sub-01_T1w.*",)
    assert image["recording"].files == 2, "the image and its sidecar"
    assert image["folder"].patterns == ("/sub-01/anat/",) and image["folder"].files == 4
    assert image["everywhere"].patterns == ("/sub-*/anat/sub-*_T1w.nii.gz",)
    assert image["everywhere"].files == 2
    assert "extension" not in image, "never every .nii.gz: that is the dataset"

    session = {c.key: c for c in bi.choices_for(["sub-01/ses-1/func/scanner.log"], paths)}
    assert session["everywhere"].label == "The same name in every subject and session"
    odd = {c.key: c for c in bi.choices_for(["sub-02/eeg/sub-02_task-x_eeg.xyz"], paths)}
    assert odd["extension"].patterns == ("*.xyz",) and odd["extension"].files == 2


def test_two_subjects_files_offer_one_line_for_every_subject(ds: Path) -> None:
    choices = bi.choices_for(["sub-01/anat/notes.txt", "sub-02/anat/notes.txt"], _paths(ds))
    keys = [c.key for c in choices]
    assert keys[0] == "exact" and choices[0].label == "These 2 files"
    assert "everywhere" in keys, "one sub-* line also covers subjects still to come"
    assert choices[keys.index("everywhere")].patterns == ("/sub-*/anat/notes.txt",)


def test_a_choice_that_hides_the_same_files_is_not_offered_twice(ds: Path) -> None:
    keys = [c.key for c in bi.choices_for(["sub-02/eeg/sub-02_task-x_eeg.mat"], _paths(ds))]
    assert keys.count("exact") == 1 and "everywhere" not in keys


def test_a_folder_is_offered_as_a_folder(ds: Path) -> None:
    first = bi.choices_for(["extra/"], _paths(ds))[0]
    assert (first.label, first.patterns, first.files) == ("This folder", ("/extra/",), 3)


def test_stop_ignoring_names_the_line_and_what_else_it_hides(ds: Path) -> None:
    (ds / ".bidsignore").write_text("/sub-01/anat/notes.txt\n*.txt\n")
    paths = _paths(ds)
    stop = bi.stop_choices(["sub-02/anat/notes.txt"], paths, ds)
    assert [(c.key, c.label) for c in stop] == [
        ("stop", "Stop ignoring this file"),
        ("remove", "Remove *.txt, which hides 4 files")]
    assert stop[0].exceptions == ("!/sub-02/anat/notes.txt",), "the rest of *.txt stays"
    assert bi.remove_patterns(ds, ["*.txt"]) == ["*.txt"]
    assert (ds / ".bidsignore").read_text() == "/sub-01/anat/notes.txt\n"


def test_several_files_stop_being_ignored_in_one_step(ds: Path) -> None:
    """Reported 2026-10-20: files ignored together had to be brought back one
    by one."""
    paths = _paths(ds)
    picked = ["sub-01/anat/notes.txt", "sub-02/anat/notes.txt", "extra/a.txt"]
    bi.apply_choice(ds, bi.choices_for(picked, paths)[0])
    assert set(bi.ignored_by(ds, paths)) == set(picked)
    stop = bi.stop_choices(picked, paths, ds)
    assert [c.label for c in stop] == ["Stop ignoring these 3 files"]
    added, removed = bi.apply_choice(ds, stop[0])
    assert (added, len(removed)) == ([], 3)
    assert bi.ignored_by(ds, paths) == {}
    assert (ds / ".bidsignore").read_text() == "\n"
    assert peek_undo(ds).label == "Stop ignoring 3 files"
    undo_last(ds)
    assert set(bi.ignored_by(ds, paths)) == set(picked), "one Undo puts all three back"


def test_a_wider_line_gets_exceptions_and_keeps_the_rest(ds: Path) -> None:
    from bidsmgr.vendor.bidsval import validate

    (ds / ".bidsignore").write_text("/sub-01/anat/notes.txt\n*.txt\n")
    paths = _paths(ds)
    picked = ["sub-01/anat/notes.txt", "extra/a.txt"]
    choice = bi.stop_choices(picked, paths, ds)[0]
    assert choice.patterns == ("/sub-01/anat/notes.txt",), "the line that hid only it"
    assert choice.exceptions == ("!/extra/a.txt", "!/sub-01/anat/notes.txt")
    bi.apply_choice(ds, choice)
    assert set(bi.ignored_by(ds, paths)) == {"sub-02/anat/notes.txt", "extra/b.txt"}
    validated = {Path(v.path).as_posix() for v in validate(ds, read_headers=False).files}
    assert set(picked) <= validated and "extra/b.txt" not in validated, \
        "the validator honours the exceptions"


def test_no_exception_for_a_file_in_an_ignored_folder(ds: Path) -> None:
    """As gitignore and the reference validator: a file inside an ignored
    folder cannot be brought back by a later ! line, so none is written."""
    from bidsmgr.vendor.bidsval.files.bidsignore import BidsIgnore

    assert BidsIgnore(["/extra/", "!/extra/a.txt"], include_defaults=False).match("extra/a.txt")
    assert not BidsIgnore(["*.txt", "!/extra/a.txt"],
                          include_defaults=False).match("extra/a.txt")
    (ds / ".bidsignore").write_text("/extra/\n")
    stop = bi.stop_choices(["extra/a.txt"], _paths(ds), ds)
    assert [(c.key, c.label) for c in stop] == [("remove", "Remove /extra/, which hides 3 files")]


def test_a_folder_picked_comes_back_whole(ds: Path) -> None:
    (ds / ".bidsignore").write_text("/extra/\n*.txt\n")
    paths = _paths(ds)
    stop = bi.stop_choices(["extra/"], paths, ds)
    assert stop[0].label == "Stop ignoring these 3 files"
    assert stop[0].patterns == ("/extra/",)
    assert stop[0].exceptions == ("!/extra/a.txt", "!/extra/b.txt")
    bi.apply_choice(ds, stop[0])
    assert not any(p.startswith("extra/") for p in bi.ignored_by(ds, paths))
    assert "sub-01/anat/notes.txt" in bi.ignored_by(ds, paths), "*.txt still hides the rest"


def test_each_change_is_one_undoable_step(ds: Path) -> None:
    assert bi.add_patterns(ds, ["/extra/"]) == ["/extra/"]
    assert peek_undo(ds).label == "Ignore in validation: /extra/"
    assert bi.add_patterns(ds, ["/extra/"]) == [], "already there: nothing written"
    assert bi.remove_patterns(ds, ["/extra/"]) == ["/extra/"]
    assert peek_undo(ds).label == "Stop ignoring: /extra/"
    undo_last(ds)
    assert (ds / ".bidsignore").read_text() == "/extra/\n"
    undo_last(ds)
    assert not (ds / ".bidsignore").exists(), "created by the first step, gone with it"


def test_the_lines_say_what_they_do(ds: Path) -> None:
    (ds / ".bidsignore").write_text("# notes\n*.log\nsourcedata/\n*.nope\n!/sub-01/ses-1/func/scanner.log\n")
    lines = {p.raw: p for p in bi.read_patterns(ds, _paths(ds))}
    assert lines["# notes"].comment
    assert lines["*.log"].count == 2
    assert lines["sourcedata/"].is_redundant and not lines["sourcedata/"].is_dead
    assert lines["*.nope"].is_dead
    assert lines["!/sub-01/ses-1/func/scanner.log"].negate
    assert bi.ignored_by(ds, _paths(ds)) == {"sub-02/ses-2/func/scanner.log": "*.log"}, \
        "sub-01's is brought back by the last line"


def test_validating_an_ignored_file_alone_reports_nothing(ds: Path) -> None:
    from bidsmgr.vendor.bidsval import validate_file

    before = validate_file(ds, "extra/a.txt", read_headers=False)
    assert before.issues, "not BIDS, and said so"
    bi.add_patterns(ds, ["/extra/"])
    assert validate_file(ds, "extra/a.txt", read_headers=False).issues == []


def test_hidden_findings_are_counted_against_a_report(ds: Path) -> None:
    from bidsmgr.vendor.bidsval import validate

    report = validate(ds, read_headers=False)
    extra = sum(len(v.issues) for v in report.files
                if Path(v.path).as_posix().startswith("extra/"))
    assert extra
    bi.add_patterns(ds, ["/extra/"])
    assert bi.findings_hidden_by(report, bi.ignored_by(ds, _paths(ds))) == extra

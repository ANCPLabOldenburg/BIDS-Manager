"""Every shape a rename can take, checked against the same three properties.

A rename is not one operation, it is a family: whole subject or part of one,
with sessions or without, into a fresh name or into an existing subject. Each
combination moves a different set of files and each has its own way of leaving
the dataset inconsistent, which is why they are enumerated here rather than
sampled.

Three properties are asserted for every one of them, because between them they
catch everything that was reported broken:

1. **No empty directory is left behind.** An empty ``anat/`` claims a modality
   that is not there; an empty ``sub-001/`` claims a subject.
2. **No scans row points at a file that is not there.** The ``filename``
   column is relative to the directory holding the table, so a row that does
   not move with its file becomes a dangling reference.
3. **Every recording is described by some scans table.** A subject created by a
   split had no table at all, so its recordings were described by nothing.

The bug that made this file necessary: scans tables live at the SESSION level
when a dataset has sessions, and the code only ever looked at the subject
level. Every session-based dataset was handled by accident or not at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import rename as rn

DATATYPES = (("anat", "T1w"), ("func", "task-rest_bold"))


def _build(
    root: Path, *, sessions: bool = True, n_ses: int = 2,
    subjects: tuple[str, ...] = ("sub-001",), stray: bool = False,
) -> Path:
    """A dataset with its scans tables where BIDS actually puts them."""
    root = root.resolve()
    for index, sub in enumerate(subjects):
        names = (
            [f"ses-{index * 10 + j + 1:02d}" for j in range(n_ses)]
            if sessions else [None]
        )
        for session in names:
            base = root / sub / session if session else root / sub
            stem = f"{sub}_{session}" if session else sub
            for datatype, suffix in DATATYPES:
                folder = base / datatype
                folder.mkdir(parents=True)
                (folder / f"{stem}_{suffix}.nii.gz").write_text("x")
                (folder / f"{stem}_{suffix}.json").write_text("{}")
            (base / f"{stem}_scans.tsv").write_text(
                "filename\tacq_time\n"
                + "".join(
                    f"{datatype}/{stem}_{suffix}.nii.gz\t2026-01-0{i + 1}\n"
                    for i, (datatype, suffix) in enumerate(DATATYPES)
                )
            )
        if stray:
            (root / sub / "notes.txt").write_text("lab notes")
    (root / "participants.tsv").write_text(
        "participant_id\n" + "".join(f"{s}\n" for s in subjects)
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _empty_dirs(root: Path) -> list[str]:
    return [
        str(p.relative_to(root)) for p in sorted(root.rglob("*"))
        if p.is_dir() and ".bidsmgr" not in p.parts and not any(p.iterdir())
    ]


def _tables(root: Path):
    for path in sorted(root.rglob("*_scans.tsv")):
        if ".bidsmgr" in path.parts:
            continue
        rows = [
            line.split("\t")[0]
            for line in path.read_text().splitlines()[1:] if line
        ]
        yield path, rows


def _dangling(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)} -> {name}"
        for path, rows in _tables(root) for name in rows
        if not (path.parent / name).exists()
    ]


def _undescribed(root: Path) -> list[str]:
    described = {
        (path.parent / name).resolve()
        for path, rows in _tables(root) for name in rows
        if (path.parent / name).exists()
    }
    recordings = {
        p.resolve() for p in root.rglob("*.nii.gz")
        if ".bidsmgr" not in p.parts
    }
    return sorted(
        str(p)[len(str(root)) + 1:] for p in recordings - described
    )


def _assert_coherent(root: Path) -> None:
    root = root.resolve()
    assert _empty_dirs(root) == [], "a folder was left claiming something"
    assert _dangling(root) == [], "a scans row names a file that is not there"
    assert _undescribed(root) == [], "a recording no table describes"


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

CASES = {
    "partial inside one session": dict(
        build=dict(), entity="sub", old="001", new="002",
        pick=lambda k: "/anat/" in k and "ses-01" in k,
    ),
    "a whole session out to a new subject": dict(
        build=dict(), entity="sub", old="001", new="002",
        pick=lambda k: "ses-01" in k,
    ),
    "a whole subject, with a stray file present": dict(
        build=dict(stray=True), entity="sub", old="001", new="009",
        pick=lambda k: True,
    ),
    "a whole subject": dict(
        build=dict(), entity="sub", old="001", new="009", pick=None,
    ),
    "partial, no sessions": dict(
        build=dict(sessions=False), entity="sub", old="001", new="002",
        pick=lambda k: "/anat/" in k,
    ),
    "a whole subject, no sessions": dict(
        build=dict(sessions=False), entity="sub", old="001", new="009",
        pick=None,
    ),
    "a session rename": dict(
        build=dict(), entity="ses", old="01", new="baseline", pick=None,
    ),
    "a task rename": dict(
        build=dict(), entity="task", old="rest", new="resting", pick=None,
    ),
    "merging a whole subject into an existing one": dict(
        build=dict(subjects=("sub-001", "sub-002"), n_ses=1),
        entity="sub", old="001", new="002", pick=None, fuse=True,
    ),
    "merging part of a subject into an existing one": dict(
        build=dict(subjects=("sub-001", "sub-002"), n_ses=1),
        entity="sub", old="001", new="002",
        pick=lambda k: "/anat/" in k, fuse=True,
    ),
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_the_dataset_stays_coherent(label: str, tmp_path: Path) -> None:
    case = CASES[label]
    root = _build(tmp_path / "ds", **case["build"])
    plan = rn.plan_rename(
        root, case["entity"], case["old"], case["new"],
        fuse=case.get("fuse", False),
    )
    assert not plan.conflicts, plan.conflicts
    pick = case["pick"]
    only = (
        None if pick is None
        else {k for k in plan.file_keys(root) if pick(k)}
    )
    touched, errors = rn.apply_rename(root, plan, only=only)
    assert not errors, errors
    _assert_coherent(root)


@pytest.mark.parametrize("label", sorted(CASES))
def test_every_case_is_one_undo(label: str, tmp_path: Path) -> None:
    from bidsmgr.project.operations import read_log, undo_last

    case = CASES[label]
    root = _build(tmp_path / "ds", **case["build"])

    def snapshot() -> dict:
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".bidsmgr" not in p.parts
        }

    before = snapshot()
    plan = rn.plan_rename(
        root, case["entity"], case["old"], case["new"],
        fuse=case.get("fuse", False),
    )
    pick = case["pick"]
    only = (
        None if pick is None
        else {k for k in plan.file_keys(root) if pick(k)}
    )
    rn.apply_rename(root, plan, only=only)
    assert len(read_log(root)) == 1, "one operation, however much it touched"
    undo_last(root)
    assert snapshot() == before


# ---------------------------------------------------------------------------
# Where the scans table lives
# ---------------------------------------------------------------------------


def test_the_table_is_found_at_the_session_level(tmp_path: Path) -> None:
    """The bug in one line: it was only ever looked for at the subject level,
    so every dataset with sessions was handled by accident."""
    root = _build(tmp_path / "ds")
    recording = (
        root / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_T1w.nii.gz"
    )
    assert rn.scans_home(recording, root) == root / "sub-001" / "ses-01"
    assert rn.scans_table_in(rn.scans_home(recording, root)).name == \
        "sub-001_ses-01_scans.tsv"


def test_the_table_is_found_at_the_subject_level_without_sessions(
    tmp_path: Path,
) -> None:
    root = _build(tmp_path / "ds", sessions=False)
    recording = root / "sub-001" / "anat" / "sub-001_T1w.nii.gz"
    assert rn.scans_home(recording, root) == root / "sub-001"
    assert rn.scans_table_in(rn.scans_home(recording, root)).name == \
        "sub-001_scans.tsv"


def test_a_path_outside_any_subject_has_no_table(tmp_path: Path) -> None:
    root = _build(tmp_path / "ds")
    assert rn.scans_home(root / "participants.tsv", root) is None


def test_a_split_creates_the_table_at_the_right_level(tmp_path: Path) -> None:
    root = _build(tmp_path / "ds")
    plan = rn.plan_rename(root, "sub", "001", "002")
    rn.apply_rename(root, plan, only={
        k for k in plan.file_keys(root) if "/anat/" in k and "ses-01" in k
    })
    created = root / "sub-002" / "ses-01" / "sub-002_ses-01_scans.tsv"
    assert created.is_file(), "the new subject needs a table where BIDS puts it"
    assert not (root / "sub-002" / "sub-002_scans.tsv").exists()


# ---------------------------------------------------------------------------
# Speed, because a rename that freezes the window is not usable
# ---------------------------------------------------------------------------


def test_planning_does_not_walk_out_of_scope_directories(
    tmp_path: Path,
) -> None:
    """``derivatives/`` can be larger than the dataset. Pruning it as the walk
    descends, rather than filtering afterwards, is what makes that free."""
    root = _build(tmp_path / "ds")
    deriv = root / "derivatives" / "fmriprep" / "sub-001" / "anat"
    deriv.mkdir(parents=True)
    for i in range(50):
        (deriv / f"sub-001_desc-{i}_T1w.nii.gz").write_text("x")

    walked = rn.walk_dataset(root)
    assert not [p for p in walked if "derivatives" in p.parts]
    plan = rn.plan_rename(root, "sub", "001", "009")
    assert not [
        src for src, _dst in plan.file_moves if "derivatives" in src.parts
    ]


def test_the_walk_is_not_resolving_every_path(tmp_path: Path) -> None:
    """``resolve()`` is a syscall per path component, and it was called twice
    per file. On a 1,262-file tree that alone was 216 ms of a 294 ms plan, and
    the dialog re-planned on every keystroke."""
    root = _build(tmp_path / "ds")
    inside = root / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_T1w.json"
    assert not rn._skip(inside, root)
    assert rn._skip(root / "derivatives" / "x.json", root)
    assert rn._skip(root / ".bidsmgr" / "x.json", root)

"""The documented workflow, on the documented data, end to end.

This is the tier that was empty, and the one the last three defects belonged
to. Each of them was an END-TO-END behaviour: every unit behaved, the
composition did not, and each was found by a user rather than by a test.

* ``*_scans.tsv`` listed only NIfTI, so a session mixing MRI with EEG and MEG
  lost the recordings, and mne-bids's acquisition times were deleted;
* the vendored bidsval shipped without its schema files;
* renaming a subject left its EEG and MEG behind.

The data is the multimodal sample the documentation publishes: one
participant, 56 MB, real Siemens and GE DICOM, real EDF, real FIF. Synthetic
fixtures cannot reach this, because there is no honest synthetic DICOM, and
DICOM is where the tool does its hardest work.

Fetched once and cached outside the checkout; skipped when the machine is
offline and has no cache. See ``tests/fixtures/sample_data.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import sample_data

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# One conversion, shared by every assertion below
# ---------------------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess:
    """A CLI verb, as a real subprocess.

    Not by importing and calling ``main``: half the platform defects this tier
    exists for live in argument handling, path spelling and exit codes, and
    an in-process call sees none of them.
    """
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} exited {proc.returncode}\n"
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    return proc


@pytest.fixture(scope="module")
def converted(tmp_path_factory) -> dict:
    """scan, convert, metadata. Once per module: it is the expensive part."""
    raw = sample_data.require("multimodal")
    work = tmp_path_factory.mktemp("multimodal")
    inventory = work / "inventory.tsv"

    # The sample ships two spreadsheets, and the tutorial uses both: the PET
    # scanner did not record nine of the fields BIDS requires, and the
    # demographics are not in any DICOM. Converting without them leaves the
    # PET sidecar legitimately incomplete, which a first version of this test
    # discovered by asserting the documented "0 errors" while skipping the
    # documented step that earns it.
    sheets = raw.parent
    dose = sheets / "dose_sheet.csv"
    participants = sheets / "participants.csv"

    _run("bidsmgr.cli.scan", str(raw), str(inventory))
    convert = [
        "bidsmgr.cli.convert", str(inventory), str(work / "bids"),
        "--raw-root", str(raw),
    ]
    if dose.is_file():
        convert += ["--pet-spreadsheet", str(dose)]
    _run(*convert)

    root = work / "bids"
    if not (root / "dataset_description.json").is_file():
        candidates = [
            p for p in root.iterdir()
            if p.is_dir() and (p / "dataset_description.json").is_file()
        ]
        assert candidates, f"no dataset root produced under {root}"
        root = candidates[0]

    metadata = ["bidsmgr.cli.metadata", str(root)]
    if participants.is_file():
        metadata += ["--participants", str(participants)]
    _run(*metadata)
    return {"raw": raw, "inventory": inventory, "root": root}


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------


def test_the_scan_finds_every_modality(converted) -> None:
    """Four modalities in one folder is the shape that broke scans.tsv.

    Asserted on ``datatype``, not on ``modality``: that column holds
    the SERIES KIND (T1w, bold, fmap, scout, report, eeg, meg), which is a
    finer thing than the family and does not contain the word "mri" at all.
    """
    frame = pd.read_csv(converted["inventory"], sep="\t", dtype=str)
    datatypes = {
        str(v).lower() for v in frame["datatype"].fillna("")
    } - {""}
    assert {"eeg", "meg"} <= datatypes, sorted(datatypes)
    assert datatypes & {"anat", "func", "fmap"}, sorted(datatypes)
    assert "pet" in datatypes, sorted(datatypes)


def test_the_phoenix_report_is_excluded(converted) -> None:
    """The sample deliberately contains a PhoenixZIPReport: a DERIVED DICOM
    with no pixel data that dcm2niix cannot convert. The scanner is supposed
    to flag it and drop it from the conversion, and the only way to know it
    still does is a series that actually is one."""
    frame = pd.read_csv(converted["inventory"], sep="\t", dtype=str)
    # The scan reports it as a "report" series with no proposed datatype,
    # which is the observable outcome: nothing downstream can convert it.
    kinds = {str(v).lower() for v in frame.get("modality", [])}
    assert "report" in kinds, sorted(kinds)
    reports = frame[frame["modality"].str.lower() == "report"]
    assert reports["datatype"].fillna("").eq("").all(), (
        "the PhoenixZIPReport was given a datatype to convert into"
    )


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def test_every_modality_reaches_the_tree(converted) -> None:
    root = converted["root"]
    present = {
        p.name for p in root.rglob("*")
        if p.is_dir() and p.name in {"anat", "func", "fmap", "eeg", "meg", "pet"}
    }
    assert {"eeg", "meg"} <= present, sorted(present)
    assert present & {"anat", "func"}, "no MRI datatype folder"


def test_the_scans_table_lists_every_recording(converted) -> None:
    """The defect reported on 2026-09-14, as a test.

    A session holding MRI and EEG and MEG produced a table naming only the
    images. Nothing in the unit suite could see it: the table is written by
    the metadata engine after a conversion neither tier ran.
    """
    root = converted["root"]
    tables = sorted(root.rglob("*_scans.tsv"))
    assert tables, "no *_scans.tsv written"

    listed: set[str] = set()
    for table in tables:
        frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
        listed |= {Path(f).name for f in frame["filename"]}

    recordings = {
        p.name for p in root.rglob("*")
        if p.is_file()
        and p.parent.name in {"anat", "func", "fmap", "eeg", "meg", "pet"}
        and (p.suffix.lower() in {".edf", ".fif", ".set", ".vhdr"}
             or p.name.endswith(".nii.gz"))
    }
    assert recordings, "no recordings in the converted tree"
    missing = recordings - listed
    assert not missing, f"not listed in any scans table: {sorted(missing)}"


def test_the_eeg_and_meg_are_in_the_table(converted) -> None:
    """Named explicitly, because "every recording" would still pass if the
    conversion had quietly produced no EEG at all."""
    root = converted["root"]
    listed: set[str] = set()
    for table in root.rglob("*_scans.tsv"):
        frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
        listed |= set(frame["filename"])
    assert any(f.endswith(".edf") for f in listed), sorted(listed)
    assert any(f.endswith(".fif") for f in listed), sorted(listed)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _errors(root) -> list[tuple[str, str, str]]:
    """Every error, with its path spelled POSIX.

    as_posix(), not str(). The callers below filter on "/pet/", and on Windows
    str(v.path) is "sub-002\\pet\\sub-002_pet.json", so the filter matched
    nothing: the PET errors were never excluded and the MRI/EEG/MEG assertion
    failed carrying a list of PET findings. CROSS_PLATFORM_RULES section 1.1,
    in a test written two commits after it.
    """
    from bidsmgr.editor.validator import validate

    report = validate(root)
    return [
        (v.path.as_posix(), i.rule_id, i.field or "")
        for v in report.files for i in (v.issues or [])
        if i.severity.value == "err" and not getattr(i, "mirrored", False)
    ]


def test_everything_but_pet_validates_clean(converted) -> None:
    """MRI, EEG and MEG convert with no errors. PET is excluded here and
    covered by the xfail below, so a regression in the other three is not
    hidden behind a known PET gap."""
    errors = [e for e in _errors(converted["root"]) if "/pet/" not in e[0]]
    assert not errors, errors


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the published dose_sheet.csv is LONG (field,value,units,...) and "
        "read_pet_spreadsheet expects WIDE (one row per scan, an identifying "
        "column). The file is rejected with 'no identifying column', so the "
        "nine fields it carries never reach the sidecar and the validator "
        "reports exactly those nine as missing. Found 2026-09-14. Either the "
        "reader learns the long form or the published sheet is regenerated; "
        "until then the multimodal tutorial's closing 'Validate: 0 errors' "
        "is not reproducible by following it."
    ),
)
def test_the_result_validates_with_no_errors(converted) -> None:
    assert not _errors(converted["root"])


def test_the_pet_gap_is_exactly_the_dose_sheet(converted) -> None:
    """Pin WHICH fields are missing, so the xfail above cannot quietly start
    covering some other PET defect."""
    missing = {e[2] for e in _errors(converted["root"]) if "/pet/" in e[0]}
    if not missing:
        pytest.skip("the PET gap is closed; tighten the xfail above")
    sheet = {
        "InjectedMass", "InjectedMassUnits", "SpecificRadioactivity",
        "SpecificRadioactivityUnits", "ModeOfAdministration",
        "AcquisitionMode", "ReconMethodParameterLabels",
        "ReconMethodParameterUnits", "ReconMethodParameterValues",
    }
    assert missing <= sheet, f"missing more than the dose sheet holds: {missing - sheet}"


def test_the_dataset_description_is_complete(converted) -> None:
    import json

    described = json.loads(
        (converted["root"] / "dataset_description.json").read_text()
    )
    for key in ("Name", "BIDSVersion", "DatasetType"):
        assert described.get(key), f"{key} is empty"


# ---------------------------------------------------------------------------
# Renaming, the other half of the scans defect
# ---------------------------------------------------------------------------


def test_renaming_a_subject_carries_every_modality(converted, tmp_path) -> None:
    """A rename is an end-to-end operation over a converted tree, so this is
    the only tier that can test it against real output."""
    import shutil

    from bidsmgr.editor.rename import apply_rename, plan_rename

    root = tmp_path / "renamed"
    shutil.copytree(converted["root"], root)

    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    assert subjects, "no subjects to rename"
    old = subjects[0].removeprefix("sub-")

    plan = plan_rename(root, "sub", old, "999")
    _touched, errors = apply_rename(root, plan)
    assert not errors, errors

    assert (root / "sub-999").is_dir()
    assert not (root / f"sub-{old}").exists(), "the old folder was left behind"

    tables = sorted((root / "sub-999").rglob("*_scans.tsv"))
    assert tables, "the renamed subject has no scans table"
    for table in tables:
        frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
        for name in frame["filename"]:
            assert f"sub-{old}" not in name, f"stale name in {table.name}: {name}"


def test_a_session_can_be_removed_and_put_back(converted, tmp_path) -> None:
    """Restructuring is an end-to-end operation over a converted tree, so this
    is the only tier that can test it against real output.

    Real data is what caught the defect this covers: the planner and the
    applier disagreed about how a path was SPELLED, and the result was a
    removed session that still existed, holding empty datatype folders and a
    scans table describing files that had moved out of it. Every unit test
    passed, because pytest's tmp_path is already canonical on macOS.

    A round trip rather than one direction, because the two halves have to
    agree: what comes out of a session has to be able to go back into one.
    """
    import shutil

    from bidsmgr.editor import restructure as rs
    from bidsmgr.editor.rename import apply_rename

    root = tmp_path / "restructured"
    shutil.copytree(converted["root"], root)

    subject = next(p for p in sorted(root.glob("sub-*")) if p.is_dir())
    sessions = sorted(p.name for p in subject.glob("ses-*") if p.is_dir())
    if not sessions:
        pytest.skip("the converted sample has no session to remove")
    before = _recordings(root)

    # Out of the session.
    _touched, errors = apply_rename(
        root, rs.plan_remove_session(root, [subject]),
    )
    assert not errors, errors
    assert not (subject / sessions[0]).exists(), (
        "the session folder was left behind"
    )
    assert (subject / f"{subject.name}_scans.tsv").is_file(), (
        "the scans table did not follow its recordings up to the subject"
    )
    assert not [n for n in _recordings(root) if sessions[0] in n]

    # And back into one.
    _touched, errors = apply_rename(
        root, rs.plan_add_session(root, "redo", [subject]),
    )
    assert not errors, errors
    table = subject / "ses-redo" / f"{subject.name}_ses-redo_scans.tsv"
    assert table.is_file(), "the table did not follow into the session"
    assert not (subject / f"{subject.name}_scans.tsv").exists()

    # Same recordings, every modality, none lost or duplicated on the way.
    after = _recordings(root)
    assert len(after) == len(before), (
        f"{len(before)} recording(s) before, {len(after)} after"
    )
    assert {_suffix_of(n) for n in after} == {_suffix_of(n) for n in before}

    # And every row in the table points at something that is there.
    frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
    for name in frame["filename"]:
        assert (table.parent / name).exists(), (
            f"scans row points at a file that does not exist: {name}"
        )


def test_an_entity_can_be_added_across_a_real_tree(converted, tmp_path) -> None:
    """Adding an entity has to place it where the STANDARD puts it, and carry
    every companion the conversion produced.

    On real output that means the sidecar, and for EEG and MEG the
    ``_channels.tsv`` and ``_events.tsv`` the converter wrote beside the
    recording. A synthetic fixture can be made to have those; only real
    output proves the conversion's own naming lines up with what the
    restructurer expects.
    """
    import shutil

    from bidsmgr.editor import restructure as rs
    from bidsmgr.editor.rename import apply_rename

    root = tmp_path / "entity-added"
    shutil.copytree(converted["root"], root)

    anat = sorted(root.rglob("*_T1w.nii.gz"))
    if not anat:
        pytest.skip("the converted sample has no T1w to work on")
    target = anat[0]
    companions = {p.name for p in rs.companions(root, target)}
    assert len(companions) >= 2, (
        f"a converted T1w should have at least a sidecar, got {companions}"
    )

    plan = rs.plan_entity_edit(root, "acq", [target], value="restructured")
    assert not plan.conflicts, plan.conflicts
    _touched, errors = apply_rename(root, plan)
    assert not errors, errors

    landed = sorted(root.rglob("*acq-restructured*"))
    assert len(landed) == len(companions), (
        "every companion should have travelled with the recording"
    )
    for path in landed:
        stem = path.name.split(".")[0]
        parts = stem.split("_")
        assert parts[0].startswith("sub-"), stem
        # acq must sit before the suffix and after sub/ses, which is the
        # schema's order, not the order it was typed in.
        assert any(p.startswith("acq-") for p in parts[:-1]), stem
        assert not parts[-1].startswith("acq-"), (
            f"acq landed in the suffix position: {stem}"
        )


def _recordings(root: Path) -> list[str]:
    """Every data file in the tree, by name, excluding tables and our state."""
    skip = ("_scans.tsv", "_channels.tsv", "_events.tsv", ".json")
    return sorted(
        p.name for p in root.rglob("*")
        if p.is_file()
        and ".bidsmgr" not in p.parts
        and p.parent.name not in ("", root.name)
        and not p.name.endswith(skip)
    )


def _suffix_of(name: str) -> str:
    return name.split(".")[0].split("_")[-1]

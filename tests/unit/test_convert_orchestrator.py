"""Unit tests for ``cli/convert.py`` orchestration.

dcm2niix is mocked at the ``subprocess.run`` boundary inside
``converter/backends/dcm2niix_direct``; these tests drive the CLI
end-to-end through ``run_convert``, exercising the three phases,
``--dataset`` filtering, ``--overwrite``, ``--dry-run``, and error-log
behavior. No real DICOMs / no real dcm2niix.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from bidsmgr.cli import convert as convert_mod
from bidsmgr.cli.convert import run_convert
from bidsmgr.converter.backends import dcm2niix_direct as backend_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dicoms(tmp_path: Path, name: str, n: int = 3) -> list[Path]:
    folder = tmp_path / "dicoms" / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"{name}_IMG{i:04d}.dcm"
        p.write_bytes(b"DICM")
        paths.append(p)
    return paths


def _write_inventory(
    tmp_path: Path, rows: list[dict], files_by_uid: dict[str, list[str]],
) -> Path:
    """Write a TSV + matching files_by_uid sidecar; return the TSV path."""
    df = pd.DataFrame(rows)
    tsv = tmp_path / "inventory.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    sidecar = tsv.with_suffix(tsv.suffix + ".files_by_uid.json.gz")
    with gzip.open(sidecar, "wb") as fh:
        fh.write(json.dumps(files_by_uid).encode("utf-8"))
    return tsv


def _row(
    *,
    series_uid: str,
    bids_name: str = "sub-001",
    session: str = "",
    datatype: str = "anat",
    suffix: str = "T1w",
    basename: Optional[str] = None,
    dataset: str = "study_a",
    include: str = "1",
    bids_guess_skip: str = "False",
) -> dict:
    return {
        "BIDS_name": bids_name,
        "session": session,
        "include": include,
        "series_uid": series_uid,
        "proposed_datatype": datatype,
        "proposed_basename": basename or f"{bids_name}_{suffix}",
        "bids_guess_suffix": suffix,
        "bids_guess_skip": bids_guess_skip,
        "dataset": dataset,
        "repetition_type": "isolated",
    }


def _fake_dcm2niix_writer(write_files: list[str], *, returncode: int = 0, stderr: str = ""):
    """Replacement for ``subprocess.run`` that writes synthetic outputs."""
    def runner(cmd, **kwargs):
        # Detect "-h" version probe — write nothing.
        if "-h" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="dcm2niix v1.0.20250506\n", stderr="",
            )
        out_idx = cmd.index("-o") + 1
        f_idx = cmd.index("-f") + 1
        out_dir = Path(cmd[out_idx])
        basename = cmd[f_idx]
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in write_files:
            target = out_dir / f"{basename}{suffix}"
            if suffix.endswith(".json"):
                target.write_text(json.dumps({"AcquisitionTime": "120000"}))
            else:
                target.write_bytes(b"\x1f\x8b\x08\x00")
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="", stderr=stderr)
    return runner


@pytest.fixture
def patch_subprocess(monkeypatch):
    """Default: dcm2niix succeeds with .nii.gz + .json. Tests can override."""
    monkeypatch.setattr(
        backend_mod.subprocess, "run",
        _fake_dcm2niix_writer([".nii.gz", ".json"]),
    )
    # Convert's _dcm2niix_version_string also calls subprocess.run.
    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_dcm2niix_writer([".nii.gz", ".json"]))
    return monkeypatch


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_subject_single_series(self, tmp_path: Path, patch_subprocess) -> None:
        """One T1w row → one converted file under bids_root/sub-001/anat/."""
        dicoms = _make_dicoms(tmp_path, "T1", n=3)
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"

        rc = run_convert(tsv, bids_parent, n_jobs=1)
        assert rc == 0

        target = bids_parent / "study_a" / "sub-001" / "anat"
        assert (target / "sub-001_T1w.nii.gz").exists()
        assert (target / "sub-001_T1w.json").exists()
        # dataset_description.json was written.
        assert (bids_parent / "study_a" / "dataset_description.json").exists()
        # Provenance written.
        prov = bids_parent / "study_a" / "sub-001" / ".bidsmgr" / "provenance.json"
        assert prov.exists()
        prov_data = json.loads(prov.read_text())
        assert prov_data["tasks"][0]["success"] is True
        # Staging cleaned up.
        assert not (bids_parent / "study_a" / ".tmp_bidsmgr").exists()

    def test_dataset_description_appends_on_rerun(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"

        run_convert(tsv, bids_parent, n_jobs=1)
        run_convert(tsv, bids_parent, n_jobs=1, overwrite=True)

        dd = json.loads(
            (bids_parent / "study_a" / "dataset_description.json").read_text()
        )
        assert isinstance(dd["GeneratedBy"], list)
        assert len(dd["GeneratedBy"]) == 2  # one entry per run

    def test_provenance_has_dcm2niix_version(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        run_convert(tsv, bids_parent, n_jobs=1)
        prov = json.loads(
            (bids_parent / "study_a" / "sub-001" / ".bidsmgr" / "provenance.json").read_text()
        )
        assert "dcm2niix" in prov["dcm2niix_version"].lower()
        assert prov["bidsmgr_version"]


# ---------------------------------------------------------------------------
# Filtering & dry-run
# ---------------------------------------------------------------------------


class TestFilteringAndDryRun:
    def test_dataset_filter_processes_only_matching_rows(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms_a = _make_dicoms(tmp_path, "T1a")
        dicoms_b = _make_dicoms(tmp_path, "T1b")
        tsv = _write_inventory(
            tmp_path,
            [
                _row(series_uid="UID_A", basename="sub-001_T1w", dataset="study_a"),
                _row(series_uid="UID_B", basename="sub-002_T1w", bids_name="sub-002", dataset="study_b"),
            ],
            {"UID_A": [str(p) for p in dicoms_a],
             "UID_B": [str(p) for p in dicoms_b]},
        )
        bids_parent = tmp_path / "out"

        run_convert(tsv, bids_parent, n_jobs=1, dataset="study_a")

        assert (bids_parent / "study_a" / "sub-001").exists()
        assert not (bids_parent / "study_b").exists()

    def test_two_datasets_split_when_no_filter(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms_a = _make_dicoms(tmp_path, "T1a")
        dicoms_b = _make_dicoms(tmp_path, "T1b")
        tsv = _write_inventory(
            tmp_path,
            [
                _row(series_uid="UID_A", basename="sub-001_T1w", dataset="study_a"),
                _row(series_uid="UID_B", basename="sub-002_T1w", bids_name="sub-002", dataset="study_b"),
            ],
            {"UID_A": [str(p) for p in dicoms_a],
             "UID_B": [str(p) for p in dicoms_b]},
        )
        bids_parent = tmp_path / "out"
        run_convert(tsv, bids_parent, n_jobs=1)
        assert (bids_parent / "study_a" / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
        assert (bids_parent / "study_b" / "sub-002" / "anat" / "sub-002_T1w.nii.gz").exists()

    def test_dry_run_writes_no_files(
        self, tmp_path: Path, patch_subprocess, capsys,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        run_convert(tsv, bids_parent, n_jobs=1, dry_run=True)
        assert not (bids_parent / "study_a").exists()
        captured = capsys.readouterr()
        assert "DRY:" in captured.out

    def test_skips_include_zero(self, tmp_path: Path, patch_subprocess) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w", include="0")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        rc = run_convert(tsv, bids_parent, n_jobs=1)
        assert rc == 0
        assert not (bids_parent / "study_a" / "sub-001").exists()

    def test_skips_bids_guess_skip_true(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w", bids_guess_skip="True")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        rc = run_convert(tsv, bids_parent, n_jobs=1)
        assert rc == 0
        assert not (bids_parent / "study_a" / "sub-001").exists()


# ---------------------------------------------------------------------------
# Overwrite & atomic commit
# ---------------------------------------------------------------------------


class TestAtomicCommit:
    def test_skip_when_target_exists_no_overwrite(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        # Pre-create the target.
        (bids_parent / "study_a" / "sub-001").mkdir(parents=True)
        (bids_parent / "study_a" / "sub-001" / "PRE-EXISTING").write_text("x")

        rc = run_convert(tsv, bids_parent, n_jobs=1)
        assert rc == 0
        # Pre-existing content untouched; no anat/ written.
        assert (bids_parent / "study_a" / "sub-001" / "PRE-EXISTING").exists()
        assert not (bids_parent / "study_a" / "sub-001" / "anat").exists()

    def test_overwrite_backs_up_existing(
        self, tmp_path: Path, patch_subprocess,
    ) -> None:
        dicoms = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [_row(series_uid="UID1", basename="sub-001_T1w")],
            {"UID1": [str(p) for p in dicoms]},
        )
        bids_parent = tmp_path / "out"
        old_target = bids_parent / "study_a" / "sub-001"
        old_target.mkdir(parents=True)
        (old_target / "OLD").write_text("preserved")

        run_convert(tsv, bids_parent, n_jobs=1, overwrite=True)

        # New tree present.
        assert (old_target / "anat" / "sub-001_T1w.nii.gz").exists()
        # Old content moved to backup.
        backup_root = bids_parent / "study_a" / ".bidsmgr" / "backup"
        backups = list(backup_root.glob("sub-001_*"))
        assert len(backups) == 1
        assert (backups[0] / "OLD").read_text() == "preserved"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_per_series_failure_does_not_abort_other_series(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """One T1w succeeds, one bold fails — subject still commits the T1w."""
        d1 = _make_dicoms(tmp_path, "T1")
        d2 = _make_dicoms(tmp_path, "B1")
        tsv = _write_inventory(
            tmp_path,
            [
                _row(series_uid="UID_T1", basename="sub-001_T1w"),
                _row(series_uid="UID_B1", basename="sub-001_task-rest_bold",
                     datatype="func", suffix="bold"),
            ],
            {"UID_T1": [str(p) for p in d1], "UID_B1": [str(p) for p in d2]},
        )
        bids_parent = tmp_path / "out"

        # Make dcm2niix succeed for T1 but fail for bold.
        def selective_runner(cmd, **kwargs):
            if "-h" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="dcm2niix v1\n", stderr="")
            f_idx = cmd.index("-f") + 1
            basename = cmd[f_idx]
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            if "T1w" in basename:
                (out_dir / f"{basename}.nii.gz").write_bytes(b"\x1f\x8b")
                (out_dir / f"{basename}.json").write_text("{}")
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="bold failed")

        monkeypatch.setattr(backend_mod.subprocess, "run", selective_runner)
        monkeypatch.setattr(convert_mod.subprocess, "run", selective_runner)

        rc = run_convert(tsv, bids_parent, n_jobs=1)
        # rc == 0 because the per-series failure is captured in
        # ConvertResult; the subject as a whole still committed.
        assert rc == 0
        # T1w landed.
        assert (bids_parent / "study_a" / "sub-001" / "anat" / "sub-001_T1w.nii.gz").exists()
        # bold did not land.
        assert not (bids_parent / "study_a" / "sub-001" / "func").exists()
        # Provenance records both attempts.
        prov = json.loads(
            (bids_parent / "study_a" / "sub-001" / ".bidsmgr" / "provenance.json").read_text()
        )
        successes = [t for t in prov["tasks"] if t["success"]]
        failures = [t for t in prov["tasks"] if not t["success"]]
        assert len(successes) == 1 and successes[0]["suffix"] == "T1w"
        assert len(failures) == 1 and failures[0]["suffix"] == "bold"

    def test_missing_files_by_uid_sidecar_raises(
        self, tmp_path: Path,
    ) -> None:
        # Write TSV but NOT the sidecar.
        tsv = tmp_path / "inv.tsv"
        pd.DataFrame([_row(series_uid="UID1")]).to_csv(tsv, sep="\t", index=False)
        with pytest.raises(FileNotFoundError, match="files_by_uid sidecar"):
            run_convert(tsv, tmp_path / "out", n_jobs=1)

    def test_missing_dataset_column_raises(self, tmp_path: Path) -> None:
        tsv = tmp_path / "inv.tsv"
        pd.DataFrame([{"BIDS_name": "sub-001", "include": "1"}]).to_csv(
            tsv, sep="\t", index=False,
        )
        # Sidecar must exist for the loader, but the dataset check fires first.
        sidecar = tsv.with_suffix(tsv.suffix + ".files_by_uid.json.gz")
        with gzip.open(sidecar, "wb") as fh:
            fh.write(b"{}")
        with pytest.raises(ValueError, match="no `dataset` column"):
            run_convert(tsv, tmp_path / "out", n_jobs=1)

    def test_unknown_series_uid_skips_with_warning(
        self, tmp_path: Path, patch_subprocess, caplog,
    ) -> None:
        """A row whose series_uid isn't in the sidecar is logged and skipped."""
        d = _make_dicoms(tmp_path, "T1")
        tsv = _write_inventory(
            tmp_path,
            [
                _row(series_uid="GHOST", basename="sub-001_T1w"),
                _row(series_uid="UID_real", basename="sub-001_T2w", suffix="T2w"),
            ],
            {"UID_real": [str(p) for p in d]},  # GHOST not present
        )
        bids_parent = tmp_path / "out"
        rc = run_convert(tsv, bids_parent, n_jobs=1)
        assert rc == 0
        # Only T2w made it through.
        anat = bids_parent / "study_a" / "sub-001" / "anat"
        assert (anat / "sub-001_T2w.nii.gz").exists()
        assert not (anat / "sub-001_T1w.nii.gz").exists()

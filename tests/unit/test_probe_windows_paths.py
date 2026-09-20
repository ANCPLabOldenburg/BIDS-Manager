"""Three holes in the probe pass on Windows. One of them emptied the template.

The one that did it is the path budget below: a SeriesInstanceUID spent twice on
one path took dcm2niix past MAX_PATH, where it dies with a stack buffer overrun
rather than an error, and the probe returns no stats for that series. No
``_derived_fields``, so no ``converter_preview`` in the scaffold, so "Already
answered by the conversion" was empty for the MRI kinds while EEG/MEG/PET —
which the scanner answers without a probe — stayed full. Measured on the
reporter's dataset: 1 of 5 MRI rows survived before the fix, 5 of 5 after.

The other two are latent rather than observed, and are pinned here because each
fails the same silent way: a probe is advisory, so every layer above it reads
"no stats" as "nothing to say" and the scan still reports success.

Each test pins the fix rather than the symptom, and each runs on every platform:
the Windows-only conditions are simulated, so a regression shows up on CI too.

The simulation swaps the ``os`` the module under test sees, never ``os.name``
itself: ``pathlib`` reads that at class-construction time to decide between
``PosixPath`` and ``WindowsPath``, so patching it globally makes ``Path()``
raise for the rest of the session — including inside pytest's own reporting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidsmgr.classifier import dcm2niix_bidsguess
from bidsmgr.inventory import probe_convert


def _fake_os(name: str, pathext: str = ".EXE;.BAT;.CMD;.COM"):
    """An ``os`` stand-in exposing only what the code under test reads."""
    return SimpleNamespace(
        name=name, pathsep=";", environ={"PATHEXT": pathext},
    )


def _pretend_package(monkeypatch: pytest.MonkeyPatch, bin_path: Path) -> None:
    """Make ``import dcm2niix`` yield a package whose ``bin_path`` is ours."""
    monkeypatch.setitem(sys.modules, "dcm2niix", SimpleNamespace(bin_path=bin_path))
    monkeypatch.setattr(dcm2niix_bidsguess.shutil, "which", lambda _name: None)


# ---------------------------------------------------------------------------
# 1. Finding the binary at all
# ---------------------------------------------------------------------------


class TestFindDcm2niix:
    """The ``dcm2niix`` wheel ships ``dcm2niix.exe`` but points at ``dcm2niix``.

    ``dcm2niix.__init__`` builds ``bin_path`` as ``<pkgdir>/dcm2niix`` on every
    platform. That is the real file on macOS and Linux and nothing at all on
    Windows, so the packaged-binary branch never matches there and the lookup
    rests entirely on ``$PATH`` finding the ``Scripts/dcm2niix.exe`` shim.

    Latent, not the reported bug: from an activated venv the shim IS on ``$PATH``
    and everything works. It bites when the process was launched another way —
    a bundled interpreter, a shortcut, ``<venv>/Scripts/python.exe -m bidsmgr``
    — and then it is a hard ``FileNotFoundError`` that skips the BidsGuess
    classifier and the probe pass both, so a scan finishes "successfully" having
    measured nothing.
    """

    def test_finds_the_exe_the_windows_wheel_actually_ships(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        exe = tmp_path / "dcm2niix.exe"
        exe.write_text("")
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("nt"))
        _pretend_package(monkeypatch, tmp_path / "dcm2niix")

        assert dcm2niix_bidsguess.find_dcm2niix() == exe

    def test_extensionless_binary_still_wins_where_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The POSIX case must keep working exactly as before."""
        binary = tmp_path / "dcm2niix"
        binary.write_text("")
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("posix"))
        _pretend_package(monkeypatch, binary)

        assert dcm2niix_bidsguess.find_dcm2niix() == binary

    def test_windows_prefers_the_extensionless_file_when_one_is_there(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A hand-placed binary with no suffix is still honoured on Windows."""
        binary = tmp_path / "dcm2niix"
        binary.write_text("")
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("nt"))
        _pretend_package(monkeypatch, binary)

        assert dcm2niix_bidsguess.find_dcm2niix() == binary

    def test_falls_back_to_path_when_the_package_has_no_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        on_path = tmp_path / "elsewhere" / "dcm2niix.exe"
        on_path.parent.mkdir()
        on_path.write_text("")
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("nt"))
        monkeypatch.setitem(
            sys.modules, "dcm2niix",
            SimpleNamespace(bin_path=tmp_path / "nothing" / "dcm2niix"),
        )
        monkeypatch.setattr(
            dcm2niix_bidsguess.shutil, "which", lambda _name: str(on_path),
        )

        assert dcm2niix_bidsguess.find_dcm2niix() == on_path

    def test_still_raises_when_there_is_no_binary_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("nt"))
        _pretend_package(monkeypatch, tmp_path / "dcm2niix")

        with pytest.raises(FileNotFoundError):
            dcm2niix_bidsguess.find_dcm2niix()

    def test_candidates_follow_pathext_and_always_include_exe(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("nt", ".COM;.BAT"))
        names = [
            p.name
            for p in dcm2niix_bidsguess._binary_candidates(Path("pkg/dcm2niix"))
        ]
        # The wheel's suffix is added even when PATHEXT forgets it, and an odd
        # PATHEXT should not cost the user the probe. It is tried in the case
        # the wheel ships, lowercase, so that on a case-insensitive filesystem
        # the path returned names the file that is actually on disk.
        assert names == ["dcm2niix", "dcm2niix.exe", "dcm2niix.COM", "dcm2niix.BAT"]

    def test_candidates_are_just_the_path_off_windows(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(dcm2niix_bidsguess, "os", _fake_os("posix"))
        assert dcm2niix_bidsguess._binary_candidates(Path("pkg/dcm2niix")) == [
            Path("pkg/dcm2niix")
        ]


# ---------------------------------------------------------------------------
# 2. Matching the output back to the series it came from
# ---------------------------------------------------------------------------


def _write_probe_output(directory: Path, stem: str, sidecar: dict) -> None:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps(sidecar))
    (directory / f"{stem}.nii.gz").write_bytes(b"")


class TestCollectProbeStatsJoin:
    """dcm2niix does not always echo the SeriesInstanceUID back.

    On a Siemens XA30 localizer (enhanced MR, SOP class
    1.2.840.10008.5.1.4.1.1.4.1) it writes ``SeriesTime`` there instead —
    ``"110927.522000"`` for a series the inventory knows as
    ``1.3.12.2.1107...403977.0.0.0`` — and a series filed under an id no row
    carries is dropped. The staging directory holds exactly one series, which
    settles the question without having to ask dcm2niix.

    Latent, not the reported bug, and platform-neutral: the series this was
    observed on are localizers, which ``_should_probe_row`` skips anyway. It is
    a hole that would swallow a real series the day dcm2niix does this to one.
    """

    UID = "1.3.12.2.1107.5.2.43.66080.2025052611091944536403977.0.0.0"

    def test_expected_uid_wins_over_a_sidecar_that_disagrees(self, tmp_path: Path):
        _write_probe_output(
            tmp_path, "probe",
            {"SeriesInstanceUID": "110927.522000", "EchoTime": 0.00369},
        )
        stats = probe_convert.collect_probe_stats(tmp_path, expected_uid=self.UID)

        assert set(stats) == {self.UID}
        assert stats[self.UID].sidecar_fields["EchoTime"] == 0.00369
        assert stats[self.UID].n_nifti == 1

    def test_expected_uid_rescues_a_sidecar_with_no_uid_at_all(self, tmp_path: Path):
        _write_probe_output(tmp_path, "probe", {"EchoTime": 0.00369})
        stats = probe_convert.collect_probe_stats(tmp_path, expected_uid=self.UID)

        assert set(stats) == {self.UID}

    def test_multi_echo_splits_collapse_onto_the_one_series(self, tmp_path: Path):
        _write_probe_output(tmp_path, "probe_e1", {"EchoTime": 0.00519})
        _write_probe_output(tmp_path, "probe_e2", {"EchoTime": 0.00765})
        stats = probe_convert.collect_probe_stats(tmp_path, expected_uid=self.UID)

        assert set(stats) == {self.UID}
        assert stats[self.UID].n_nifti == 2
        # First sidecar still wins for the shared acquisition facts.
        assert stats[self.UID].sidecar_fields["EchoTime"] == 0.00519

    def test_without_expected_uid_it_still_groups_by_the_sidecar(self, tmp_path: Path):
        _write_probe_output(tmp_path, "a", {"SeriesInstanceUID": "uid-a"})
        _write_probe_output(tmp_path, "b", {"SeriesInstanceUID": "uid-b"})
        stats = probe_convert.collect_probe_stats(tmp_path)

        assert set(stats) == {"uid-a", "uid-b"}


# ---------------------------------------------------------------------------
# 3. Staying under MAX_PATH
# ---------------------------------------------------------------------------


class TestPathBudget:
    """A UID spent twice on one path is what killed dcm2niix on Windows.

    ``<uid>/`` as the working directory plus ``-f %j`` as the output basename
    cost ~120 characters of a 260-character ceiling, and dcm2niix does not fail
    gracefully past it: it dies with ``0xC0000409`` (stack buffer overrun) and
    an empty stderr, so the probe recorded nothing and said nothing.
    """

    UID = "1.3.12.2.1107.5.2.43.66080.2025052611091944536403977.0.0.0"

    def test_series_directory_is_short_and_stable(self):
        name = probe_convert._series_dirname(self.UID)

        assert len(name) == 12
        assert name == probe_convert._series_dirname(self.UID)
        assert name != probe_convert._series_dirname(self.UID.replace("0.0.0", "1.0.0"))

    def test_output_basename_is_short_and_not_the_uid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """``-f`` must be a short name we control, as in ``dcm2niix_direct``."""
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(probe_convert.subprocess, "run", fake_run)
        probe_convert._run_dcm2niix_full(
            tmp_path, tmp_path, dcm2niix_bin=Path("dcm2niix"),
        )

        cmd = captured["cmd"]
        basename = cmd[cmd.index("-f") + 1]
        assert basename == probe_convert.PROBE_BASENAME
        assert "%j" not in basename
        assert len(basename) < 12

    def test_probe_rows_builds_a_hashed_series_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """End to end through ``probe_rows``: no UID anywhere in the path."""
        seen: list[Path] = []

        def fake_probe_one(uid, files, series_workdir, binary):
            seen.append(Path(series_workdir))
            return uid, None

        monkeypatch.setattr(probe_convert, "_probe_one_series", fake_probe_one)
        monkeypatch.setattr(probe_convert, "find_dcm2niix", lambda: Path("dcm2niix"))

        row = SimpleNamespace(
            modality="mri", series_uid=self.UID, fine_modality="bold",
            subject_hint="001",
        )
        probe_convert.probe_rows(
            [row], tmp_path, {self.UID: ["/src/a.dcm"]}, n_jobs=1,
        )

        assert seen, "probe_rows should have queued the series"
        workdir = seen[0]
        assert self.UID not in str(workdir)
        assert workdir.name == probe_convert._series_dirname(self.UID)
        # What the probe adds under the caller's scratch root stays small, so
        # the 260-char ceiling is the caller's budget rather than ours.
        assert len(str(workdir.relative_to(tmp_path))) < 30


def test_returncode_hint_explains_the_windows_stack_overrun(tmp_path: Path):
    hint = probe_convert._returncode_hint(0xC0000409, tmp_path)
    assert "MAX_PATH" in hint and "0xC0000409" in hint
    assert probe_convert._returncode_hint(0, tmp_path) == ""
    assert probe_convert._returncode_hint(2, tmp_path) == ""


def test_os_is_imported_for_the_windows_branch():
    """``find_dcm2niix`` reads ``os.name``; a missing import would only show on
    the branch that runs on Windows, which no CI runner exercises."""
    assert dcm2niix_bidsguess.os is os

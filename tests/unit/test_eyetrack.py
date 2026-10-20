"""EyeLink eye tracking: finding, reading and converting to BIDS (BEP020)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.converter.backends import eyelink as backend
from bidsmgr.inventory import eyelink as el
from bidsmgr.inventory import eyetrack

eyelinkio = pytest.importorskip("eyelinkio")
DATA = Path(eyelinkio.__file__).parent / "tests" / "data"


@pytest.fixture
def mono(tmp_path) -> Path:
    """A monocular (left eye) recording with a pause between its two blocks."""
    return Path(shutil.copy(DATA / "test_raw.edf", tmp_path / "test_raw.edf"))


@pytest.fixture
def both(tmp_path) -> Path:
    return Path(shutil.copy(DATA / "test_raw_binocular.edf", tmp_path / "binocular.edf"))


def _european_edf(path: Path) -> Path:
    path.write_bytes(b"0       " + b" " * 248)
    return path


# ---------------------------------------------------------------------------
# Detection and reading
# ---------------------------------------------------------------------------


def test_an_eyelink_edf_is_told_from_a_european_one(mono, tmp_path) -> None:
    assert el.is_eyelink_edf(mono)
    assert not el.is_eyelink_edf(_european_edf(tmp_path / "eeg.edf"))
    pre = el.read_preamble(mono)
    assert pre["serial"] == "CL1-ACA32" and pre["tracker"].startswith("EYELINK II CL")


def test_the_tracker_clock_keeps_the_pause_between_blocks(mono) -> None:
    rec = el.read_edf_api(mono)
    assert rec.reader == "eyelinkio" and rec.eyes == ("left",) and rec.sfreq == 1000.0
    assert len(rec.blocks) == 2
    assert len(rec.times) == 66827
    gap = float(np.max(np.diff(rec.times)))
    assert gap > 40_000, "the 48 s pause is in the timestamps"
    assert rec.duration_s > 110, "the recording spans the pause, unlike eyelinkio's times"
    assert rec.gaze["left"].shape == (66827, 3)


def test_the_events_are_the_ones_eyelinkio_finds(both) -> None:
    rec = el.read_edf_api(both)
    ref = eyelinkio.read_edf(str(both))
    kinds = {k: sum(1 for e in rec.events if e.kind == k) for k in ("fixation", "saccade",
                                                                    "blink")}
    assert kinds == {"fixation": len(ref["discrete"]["fixations"]),
                     "saccade": len(ref["discrete"]["saccades"]),
                     "blink": len(ref["discrete"]["blinks"])}
    assert rec.eyes == ("left", "right") and rec.sfreq == 500.0


def test_a_probe_reads_events_without_samples(both) -> None:
    rec = el.read_edf_api(both, samples=False)
    assert len(rec.times) == 0 and rec.eyes == ("left", "right") and rec.events


def test_calibration_facts(mono) -> None:
    facts = el.calibration_facts(el.read_edf_api(mono), "left")
    assert facts["CalibrationType"] == "HV5" and facts["EyeTrackingMethod"] == "P-CR"
    assert facts["AverageCalibrationError"] == 0.29
    assert facts["MaximalCalibrationError"] == 0.65
    assert facts["PupilFitMethod"] == "centre-of-mass" and facts["CalibrationCount"] == 2


def _as_asc(rec, path: Path) -> Path:
    """The recording as edf2asc writes it (the parts the reader uses)."""
    lines = [f"** DATE: {rec.info.get('date', '')}", f"** VERSION: {rec.info.get('version')}"]
    eyes = " ".join(e.upper() for e in rec.eyes)
    lines.append(f"START\t{int(rec.times[0])} \t{eyes}\tSAMPLES\tEVENTS")
    lines.append(f"PUPIL\t{rec.pupil.upper()}")
    lines.append(f"SAMPLES\tGAZE\t{eyes}\tRATE\t{rec.sfreq:.2f}\tTRACKING\tCR")
    for t, text in rec.messages:
        lines.append(f"MSG\t{int(t)} {text}")
    for e in rec.events:
        tag = {"fixation": "EFIX", "saccade": "ESACC", "blink": "EBLINK"}[e.kind]
        lines.append(f"{tag} {e.eye[0].upper()}   {int(e.start)}\t{int(e.end)}\t"
                     f"{int(e.end - e.start)}")
    for i, t in enumerate(rec.times):
        vals = []
        for eye in rec.eyes:
            x, y, p = rec.gaze[eye][i]
            vals += ["." if np.isnan(x) else f"{x:.1f}", "." if np.isnan(y) else f"{y:.1f}",
                     f"{p:.1f}"]
        lines.append(f"{int(t)}\t" + "\t".join(vals) + "\t...")
    lines.append(f"END\t{int(rec.times[-1])} \tSAMPLES\tEVENTS")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def test_asc_reads_the_same_recording(both, tmp_path) -> None:
    rec = el.read_edf_api(both)
    asc = el.read_asc(_as_asc(rec, tmp_path / "binocular.asc"))
    assert el.is_eyelink_asc(tmp_path / "binocular.asc")
    assert asc.eyes == rec.eyes and asc.sfreq == rec.sfreq and asc.pupil == rec.pupil
    assert np.array_equal(asc.times, rec.times)
    for eye in rec.eyes:
        np.testing.assert_allclose(asc.gaze[eye], rec.gaze[eye].round(1), atol=0.051,
                                   equal_nan=True)
    assert len(asc.events) == len(rec.events) and len(asc.messages) == len(rec.messages)


def test_without_a_reader_the_reason_is_said(monkeypatch, mono) -> None:
    monkeypatch.setattr(el, "_bindings", lambda: (None, None, "not here"))
    monkeypatch.setattr(el, "edf2asc_path", lambda: None)
    reader, why = el.edf_reader()
    assert reader == "" and "not here" in why and "edf2asc" in why and ".asc" in why
    with pytest.raises(RuntimeError, match="not here"):
        el.read_recording(mono)


@pytest.mark.parametrize("platform, machine, says", [
    ("win32", "AMD64", "Visual C++ 2008"),
    ("linux", "aarch64", "x86_64 only"),
    ("linux", "x86_64", "Reinstall BIDS Manager"),
    ("darwin", "arm64", "Reinstall BIDS Manager"),
])
def test_the_remedy_fits_the_machine(platform, machine, says) -> None:
    """Measured 2026-10-12: the Windows DLL is linked against the VC++ 2008
    runtime (MSVCR90, Microsoft.VC90.CRT in its manifest); the Linux library
    exists for x86_64 only (it fails to load on aarch64)."""
    text = el.remedy(platform, machine)
    assert says in text and "edf2asc" in text
    if platform == "win32":
        assert el.VC2008_X64_URL in text


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def test_the_scan_lists_eye_tracking_and_eeg_never_sees_it(mono, both, tmp_path) -> None:
    from bidsmgr.inventory.eeg_meg import candidate_paths

    eeg = _european_edf(tmp_path / "rest.edf")
    assert candidate_paths(tmp_path) == [eeg]
    df = eyetrack.scan_eyetrack(tmp_path).set_index("source_file")
    row = df.loc["test_raw.edf"]
    assert row["modality"] == "eyetrack" and row["datatype"] == "beh"
    assert row["bids_name"].endswith("_recording-eye1_physio") and row["include"] == 1
    assert row["sfreq"] == 1000.0 and abs(float(row["duration_sec"]) - 66.8) < 0.1, \
        "what was recorded: the blocks, without the pause"
    assert row["study_date"] == "20140314" and row["acq_time"] == "140321"
    assert "binocular" in df.loc["binocular.edf"]["issues"]


def test_with_no_reader_the_row_is_listed_and_left_out(monkeypatch, mono, tmp_path) -> None:
    monkeypatch.setattr(eyetrack, "edf_reader", lambda: ("", "needs eyelinkio"))
    row = eyetrack.scan_eyetrack(tmp_path).iloc[0]
    assert row["include"] == 0 and row["bids_guess_skip"]
    assert row["issues"].startswith(eyetrack.NO_READER_TOKEN)


# ---------------------------------------------------------------------------
# Writing BIDS
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[list[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


def test_names_follow_the_eye() -> None:
    assert backend.name_for("sub-01_task-a_recording-eye1_physio", "right") == \
        "sub-01_task-a_recording-eye2_physio"
    assert backend.name_for("sub-01_task-a_physio", "left", "physioevents") == \
        "sub-01_task-a_recording-eye1_physioevents"
    assert backend.match_of("sub-01_task-a_run-1_recording-eye2_physio") == "sub-01_task-a_run-1"


def test_a_binocular_recording_writes_both_eyes(both, tmp_path) -> None:
    rec = el.read_edf_api(both)
    files = backend.write_bids(rec, tmp_path / "beh", "sub-01_task-x_recording-eye1_physio")
    names = sorted(f.name for f in files)
    assert names == sorted(f"sub-01_task-x_recording-{e}_{s}"
                           for e in ("eye1", "eye2")
                           for s in ("physio.tsv.gz", "physio.json", "physioevents.tsv.gz",
                                     "physioevents.json"))
    side = json.loads((tmp_path / "beh" / "sub-01_task-x_recording-eye2_physio.json")
                      .read_text())
    assert side["RecordedEye"] == "right" and side["PhysioType"] == "eyetrack"
    assert side["Columns"] == ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"]
    assert side["SamplingFrequency"] == 500.0 and side["StartTime"] == 0.0
    assert "area" in side["pupil_size"]["Description"]


def test_samples_keep_the_tracker_clock_and_missing_is_na(mono, tmp_path) -> None:
    rec = el.read_edf_api(mono)
    backend.write_bids(rec, tmp_path, "sub-01_task-x_recording-eye1_physio")
    rows = _rows(tmp_path / "sub-01_task-x_recording-eye1_physio.tsv.gz")
    assert len(rows) == 66827 and len(rows[0]) == 4
    stamps = np.array([int(r[0]) for r in rows])
    assert stamps[0] == int(rec.times[0]) and np.max(np.diff(stamps)) > 40_000
    assert any(r[1] == "n/a" for r in rows), "lost gaze is n/a"
    events = _rows(tmp_path / "sub-01_task-x_recording-eye1_physioevents.tsv.gz")
    assert {r[2] for r in events} >= {"fixation", "saccade", "blink", "n/a"}
    assert [int(r[0]) for r in events] == sorted(int(r[0]) for r in events)
    side = json.loads((tmp_path / "sub-01_task-x_recording-eye1_physioevents.json")
                      .read_text())
    assert side["OnsetSource"] == "timestamp"


def test_start_time_from_a_message_or_a_number(mono) -> None:
    rec = el.read_edf_api(mono)
    first_trial = rec.messages_matching("TRIALID 1")[0]
    expected = round((rec.times[0] - first_trial) / 1000.0, 6)
    assert backend.start_time(rec, {"sync_message": "TRIALID 1"}) == expected
    assert backend.start_time(rec, {"start_time": -2.5, "sync_message": "TRIALID 1"}) == -2.5
    assert backend.start_time(rec, {"sync_message": "never logged"}) == 0.0
    assert backend.start_time(rec, {}) == 0.0


def test_the_screen_goes_beside_the_runs_events_only(mono, tmp_path) -> None:
    from bidsmgr.converter.types import ConvertTask
    from bidsmgr.fixups.eyetrack import place_stimulus_presentation

    task = ConvertTask(row_id="r", series_uid="", source_files=(mono,), dataset="ds",
                       bids_root=tmp_path, subject="01", datatype="beh", suffix="physio",
                       modality="eyetrack", basename="sub-01_task-x_recording-eye1_physio",
                       eyetrack={"screen_size": [0.5, 0.3], "screen_distance": 0.6})
    beh = tmp_path / "staging" / "beh"
    beh.mkdir(parents=True)
    assert place_stimulus_presentation(tmp_path / "staging", [task]) == 0
    assert not (beh / "sub-01_task-x_events.json").exists(), "no events, no sidecar"
    (beh / "sub-01_task-x_events.tsv").write_text("onset\tduration\n1\t0\n")
    assert place_stimulus_presentation(tmp_path / "staging", [task]) == 1
    screen = json.loads((beh / "sub-01_task-x_events.json").read_text())["StimulusPresentation"]
    assert screen == {"ScreenResolution": [1920, 1080], "ScreenOrigin": ["top", "left"],
                      "ScreenSize": [0.5, 0.3], "ScreenDistance": 0.6}


def test_display_extent() -> None:
    assert el._extent(0, 1919) == 1920 and el._extent(0, 1280) == 1280
    assert el._extent(0, 1079) == 1080 and el._extent(0, 1024) == 1024


# ---------------------------------------------------------------------------
# Through a scan and a conversion
# ---------------------------------------------------------------------------


def test_scan_convert_validate(mono, both, tmp_path) -> None:
    from bidsmgr.cli.convert import run_convert
    from bidsmgr.cli.scan import run_scan
    from bidsmgr.editor.validator import validate

    raw = tmp_path / "raw"
    (raw / "a").mkdir(parents=True)
    (raw / "b").mkdir()
    shutil.move(str(mono), raw / "a" / "task-search_eyes.edf")
    shutil.move(str(both), raw / "b" / "task-search_eyes.edf")
    inv = tmp_path / "inv.tsv"
    df = run_scan(raw, inv, dataset="ds").fillna("")
    assert set(df["modality"]) == {"eyetrack"} and len(df) == 2
    assert run_convert(inv, tmp_path / "out", raw_root=raw) == 0
    out = tmp_path / "out" / "ds"
    made = sorted(p.name for p in out.rglob("*_physio.tsv.gz"))
    assert len(made) == 3, made                      # one eye + two eyes
    report = validate(out)
    errors = [i for f in report.files for i in f.issues if i.severity.value == "err"]
    # The one error left is a reported defect, not fixed here: the EyeLink
    # backend writes no events sidecar stating StimulusPresentation (pinned by
    # the strict xfail below). Anything else is new.
    assert {e.rule_id for e in errors} <= {"INCOMPLETE_STIMULUS_PRESENTATION"}, \
        [(e.rule_id, e.message) for e in errors]


@pytest.mark.xfail(strict=True, reason=(
    "Reported defect, not fixed here: the EyeLink backend writes no events "
    "sidecar with StimulusPresentation (ScreenDistance, ScreenOrigin, "
    "ScreenResolution, ScreenSize), so a converted eye tracker looking at a "
    "screen is INCOMPLETE_STIMULUS_PRESENTATION, an error for the reference "
    "validator too."))
def test_eye_tracking_states_its_screen(mono, both, tmp_path) -> None:
    from bidsmgr.cli.convert import run_convert
    from bidsmgr.cli.scan import run_scan
    from bidsmgr.editor.validator import validate

    raw = tmp_path / "raw"
    (raw / "a").mkdir(parents=True)
    shutil.move(str(mono), raw / "a" / "task-search_eyes.edf")
    inv = tmp_path / "inv.tsv"
    run_scan(raw, inv, dataset="ds")
    assert run_convert(inv, tmp_path / "out", raw_root=raw) == 0
    report = validate(tmp_path / "out" / "ds")
    codes = {i.rule_id for f in report.files for i in f.issues}
    assert "INCOMPLETE_STIMULUS_PRESENTATION" not in codes


def test_the_viewer_reads_the_clock_and_the_blinks(mono, tmp_path) -> None:
    from bidsmgr.viz.data.signal import open_physio

    rec = el.read_edf_api(mono)
    backend.write_bids(rec, tmp_path, "sub-01_task-x_recording-eye1_physio")
    src = open_physio(tmp_path / "sub-01_task-x_recording-eye1_physio.tsv.gz")
    assert src.ch_names == ["x_coordinate", "y_coordinate", "pupil_size"]
    assert src.ch_types == ["eyegaze", "eyegaze", "pupil"]
    labels = {e.label for e in src.events()}
    assert "blink" in labels and "TRIALID 1" in labels
    assert not any(label.startswith("!CAL") for label in labels)
    assert max(e.onset for e in src.events()) <= src.n_times / src.sfreq


def test_sourcedata_keeps_it_as_eye_tracking() -> None:
    from bidsmgr.converter.sourcedata import kind_for

    assert kind_for("eyetrack", "beh", "physio") == "eyetrack"
    assert kind_for("eyetrack", "func", "physio") == "eyetrack"
    assert kind_for("mri", "func", "physio") == "physio", "a scanner's physio stays physio"


def test_the_metadata_block_inherits() -> None:
    from bidsmgr.recording_meta import EyetrackSpec, RecordingMetaSpec, resolve_eyetrack

    spec = RecordingMetaSpec(
        eyetrack_defaults=EyetrackSpec(screen_size=[0.5, 0.3], screen_distance=0.6),
        eyetrack_overrides={"a.edf": EyetrackSpec(screen_distance=0.7, sync_message="GO")})
    a = resolve_eyetrack(spec, "a.edf")
    assert a.screen_size == [0.5, 0.3] and a.screen_distance == 0.7 and a.sync_message == "GO"
    assert resolve_eyetrack(spec, "b.edf").screen_distance == 0.6


def test_the_preview_converts_an_eyelink_row(tmp_path, mono) -> None:
    """REGRESSION 2026-10-12: the row preview took an EyeLink .edf for a
    European Data Format recording and handed it to the EEG reader."""
    from bidsmgr.cli.convert import preview_row
    from bidsmgr.inventory.sources import ScanSources

    raw = tmp_path / "raw" / "subjA"
    raw.mkdir(parents=True)
    shutil.move(str(mono), raw / "test_raw.edf")
    df = eyetrack.scan_eyetrack(tmp_path / "raw")
    row = df.iloc[0].to_dict()
    sources = ScanSources(tmp_path / "raw", tmp_path / "raw", {})
    shown = preview_row(row, tmp_path / "work", sources)
    names = sorted(p.name for p in shown)
    assert any(n.endswith("_physio.tsv.gz") for n in names), names
    assert not any(n.endswith(".edf") for n in names), "converted, not read as EEG"

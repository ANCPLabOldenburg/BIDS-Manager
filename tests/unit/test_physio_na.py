"""A missing physio sample is written n/a, and every reader still reads a gap.

numpy's ``savetxt`` wrote ``nan``, which a validator reads as a value of the
wrong type (TSV_VALUE_INCORRECT_TYPE on the multimodal sample's cardiac
channel, 2026-10-19). The vendored bidsphysio writes ``n/a`` now; the viewers
read ``n/a`` as the gap they read ``nan`` as.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np

from bidsmgr.vendor.bidsphysio.base import bidsphysio as bp


def _record(tmp_path: Path, signals: dict[str, np.ndarray]) -> Path:
    data = bp.PhysioData(signals=[
        bp.PhysioSignal(label=label, samples_per_second=100.0, signal=values,
                        physiostarttime=0, neuralstarttime=0)
        for label, values in signals.items()])
    prefix = tmp_path / "sub-01" / "func" / "sub-01_task-x"
    prefix.parent.mkdir(parents=True)
    data.save_bids_data(str(prefix))
    return Path(str(prefix) + "_physio.tsv.gz")


def _lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return fh.read().splitlines()


def test_a_gap_is_written_n_a(tmp_path: Path) -> None:
    cardiac = np.array([0.5, np.nan, 1.25, np.nan, np.nan, 2.0])
    out = _record(tmp_path, {"cardiac": cardiac, "respiratory": np.arange(6) / 4})
    lines = _lines(out)
    assert lines == ["0.5000\t0.0000", "n/a\t0.2500", "1.2500\t0.5000",
                     "n/a\t0.7500", "n/a\t1.0000", "2.0000\t1.2500"]
    assert not any("nan" in line for line in lines)


def test_a_long_recording_is_written_in_chunks_as_one(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=bp._ROWS_PER_CHUNK * 2 + 17)
    values[::997] = np.nan
    out = _record(tmp_path, {"cardiac": values})
    expected = io.BytesIO()
    np.savetxt(expected, values[:, None], fmt="%.4f", delimiter="\t")
    with gzip.open(out, "rb") as fh:
        assert fh.read() == expected.getvalue().replace(b"nan", b"n/a")


def test_the_viewers_read_n_a_as_a_gap(tmp_path: Path) -> None:
    from bidsmgr.viz.data.physio import read_columns

    cardiac = np.array([0.5, np.nan, 1.25, np.nan, 2.0])
    out = _record(tmp_path, {"cardiac": cardiac})
    columns, total, step = read_columns(out)
    assert (total, step) == (5, 1)
    got = columns[0]
    assert np.array_equal(np.isnan(got), np.isnan(cardiac))
    assert np.allclose(got[~np.isnan(got)], cardiac[~np.isnan(cardiac)])


def test_the_validator_finds_no_wrong_type(tmp_path: Path) -> None:
    from bidsmgr.vendor.bidsval import validate

    root = tmp_path
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "p", "BIDSVersion": "1.10.0", "Authors": ["A"]}))
    out = _record(root, {"cardiac": np.array([0.5, np.nan, 1.25, 2.0])})
    out.with_name("sub-01_task-x_physio.json").write_text(json.dumps(
        {"SamplingFrequency": 100.0, "StartTime": 0, "Columns": ["cardiac"]}))
    report = validate(root, read_headers=True)
    codes = {issue.code for verdict in report.files for issue in verdict.issues
             if Path(verdict.path).name == out.name}
    assert "TSV_VALUE_INCORRECT_TYPE" not in codes, codes

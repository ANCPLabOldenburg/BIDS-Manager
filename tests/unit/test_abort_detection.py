"""Unit tests for ``cli.scan._detect_aborts`` and ``_has_design_marker``.

Heuristic (architecture.md §4.1, abort detection):

* ``isolated`` — singleton group.
* ``trivial`` — n_files ≤ 2 AND a same-name companion in the group has
  at least 10× more files (likely Phoenix mosaic / MoCo summary /
  derivative output).
* ``planned`` — intentional repeat (default for non-trivial members of a
  multi-row group).
* ``suspected_abort`` — same SeriesDescription AND same image_type as a
  later companion within the redo window, both sides have ≥ 10 files,
  and no design marker in the name. Operator likely re-recorded after a
  noisy initial attempt; the EARLIER attempt is flagged.
"""

from __future__ import annotations

from pathlib import Path

from bidsmgr.classifier.types import Classification
from bidsmgr.cli.scan import _detect_aborts, _has_design_marker, _normalize_runs
from bidsmgr.inventory.types import InventoryRow


def _row(
    *,
    n_files=100,
    acq_time="120000",
    subject="001",
    session="pre",
    series_description="task-foo_bold",
    image_type="M",
) -> InventoryRow:
    return InventoryRow(
        modality="mri",
        source=Path("/tmp/x"),
        subject_hint=subject,
        session_hint=session,
        n_files=n_files,
        acq_time=acq_time,
        series_description=series_description,
        image_type=image_type,
    )


def _cls(row: InventoryRow, **entities) -> Classification:
    return Classification(
        row_id=row.row_id, classifier="test",
        datatype="func", suffix="bold",
        candidate_entities={"task": "foo", **entities},
        confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Design-marker detector
# ---------------------------------------------------------------------------


def test_design_marker_detects_run():
    assert _has_design_marker("task-foo_run-01_bold")
    assert _has_design_marker("task-foo_run-1_bold")
    assert _has_design_marker("task-foo_run01_bold")


def test_design_marker_detects_split_part():
    assert _has_design_marker("task-foo_split-1_bold")
    assert _has_design_marker("task-foo_part-2_bold")


def test_design_marker_no_match_on_name_without_index():
    assert not _has_design_marker("cmrr_mbep2d_1p25_fMRI_PA")
    assert not _has_design_marker("nc_epi3d_v3u_1seg_96part_AP")
    assert not _has_design_marker("MPRAGE")


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_singleton_is_isolated():
    r = _row()
    chosen = {r.row_id.hex: _cls(r)}
    verdicts = _detect_aborts([r], chosen)
    assert verdicts[r.row_id.hex] == "isolated"


def test_design_marker_in_name_skips_abort_detection():
    """Operator-encoded ``run-N`` is intent — never flag as abort."""
    r1 = _row(series_description="task-foo_run-01_bold", n_files=200, acq_time="120000")
    r2 = _row(series_description="task-foo_run-02_bold", n_files=200, acq_time="120030")
    chosen = {r1.row_id.hex: _cls(r1), r2.row_id.hex: _cls(r2)}
    verdicts = _detect_aborts([r1, r2], chosen)
    assert verdicts[r1.row_id.hex] == "planned"
    assert verdicts[r2.row_id.hex] == "planned"


def test_two_complete_attempts_close_in_time_flag_earlier_as_abort():
    """The user's primary case: technician redo after noisy first attempt."""
    r1 = _row(n_files=200, acq_time="120000", image_type="M")
    r2 = _row(n_files=200, acq_time="120030", image_type="M")  # 30s later, same name+type
    chosen = {r1.row_id.hex: _cls(r1), r2.row_id.hex: _cls(r2)}
    verdicts = _detect_aborts([r1, r2], chosen)
    assert verdicts[r1.row_id.hex] == "suspected_abort"
    assert verdicts[r2.row_id.hex] == "planned"


def test_magnitude_phase_pair_is_NOT_abort():
    """One acquisition outputting M and P should not be flagged as redo of itself.

    This was the false-positive reported by the user — different image_type
    means different physical sub-output of the same acquisition.
    """
    mag = _row(n_files=200, acq_time="120000", image_type="M")
    phase = _row(n_files=200, acq_time="120030", image_type="P")
    chosen = {mag.row_id.hex: _cls(mag), phase.row_id.hex: _cls(phase)}
    verdicts = _detect_aborts([mag, phase], chosen)
    assert verdicts[mag.row_id.hex] == "planned"
    assert verdicts[phase.row_id.hex] == "planned"


def test_trivial_one_file_with_large_same_name_companion():
    """1-file Phoenix mosaic next to 132-file actual run → trivial."""
    derivative = _row(n_files=1, acq_time="120000", image_type="M")
    actual = _row(n_files=132, acq_time="120020", image_type="M")
    chosen = {derivative.row_id.hex: _cls(derivative), actual.row_id.hex: _cls(actual)}
    verdicts = _detect_aborts([derivative, actual], chosen)
    assert verdicts[derivative.row_id.hex] == "trivial"
    assert verdicts[actual.row_id.hex] == "planned"
    # And the trivial row never triggers an abort flag on the longer companion.
    assert verdicts[actual.row_id.hex] != "suspected_abort"


def test_two_file_sbref_alone_is_NOT_trivial():
    """SBRef genuinely has 1-2 files; without a much larger same-name companion
    it must stay ``planned`` (not ``trivial``) so the GUI keeps it convertible.
    """
    sbref1 = _row(n_files=2, acq_time="120000", image_type="M",
                  series_description="task-foo_sbref")
    sbref2 = _row(n_files=2, acq_time="121000", image_type="M",
                  series_description="task-foo_sbref")
    chosen = {
        sbref1.row_id.hex: Classification(
            row_id=sbref1.row_id, classifier="test", datatype="func", suffix="sbref",
            candidate_entities={"task": "foo"}, confidence=0.5,
        ),
        sbref2.row_id.hex: Classification(
            row_id=sbref2.row_id, classifier="test", datatype="func", suffix="sbref",
            candidate_entities={"task": "foo"}, confidence=0.5,
        ),
    }
    verdicts = _detect_aborts([sbref1, sbref2], chosen)
    assert verdicts[sbref1.row_id.hex] != "trivial"
    assert verdicts[sbref2.row_id.hex] != "trivial"


def test_short_acquisition_far_in_time_is_planned_not_abort():
    """5 minutes is outside the redo window — different planned attempts."""
    r1 = _row(n_files=200, acq_time="120000", image_type="M")
    r2 = _row(n_files=200, acq_time="121000", image_type="M")  # 10 min later
    chosen = {r1.row_id.hex: _cls(r1), r2.row_id.hex: _cls(r2)}
    verdicts = _detect_aborts([r1, r2], chosen)
    assert verdicts[r1.row_id.hex] == "planned"
    assert verdicts[r2.row_id.hex] == "planned"


def test_normalize_runs_excludes_aborts_and_trivials():
    """Aborts and trivials must not consume a ``run-N`` number."""
    abort = _row(n_files=200, acq_time="120000", image_type="M")
    redo = _row(n_files=200, acq_time="120030", image_type="M")
    derivative = _row(n_files=1, acq_time="120000", image_type="M")
    rows = [abort, redo, derivative]
    chosen = {r.row_id.hex: _cls(r) for r in rows}
    verdicts = _detect_aborts(rows, chosen)
    assert verdicts[abort.row_id.hex] == "suspected_abort"
    assert verdicts[derivative.row_id.hex] == "trivial"

    chosen = _normalize_runs(rows, chosen, verdicts)
    # redo stands alone in run-numbering → no run entity.
    assert "run" not in chosen[abort.row_id.hex].candidate_entities
    assert "run" not in chosen[derivative.row_id.hex].candidate_entities
    assert "run" not in chosen[redo.row_id.hex].candidate_entities


def test_three_complete_attempts_only_first_two_are_aborts():
    """Operator runs three times within window → first two are aborts, last is kept."""
    a1 = _row(n_files=200, acq_time="120000", image_type="M")
    a2 = _row(n_files=200, acq_time="120100", image_type="M")
    a3 = _row(n_files=200, acq_time="120200", image_type="M")
    chosen = {r.row_id.hex: _cls(r) for r in [a1, a2, a3]}
    verdicts = _detect_aborts([a1, a2, a3], chosen)
    assert verdicts[a1.row_id.hex] == "suspected_abort"
    assert verdicts[a2.row_id.hex] == "suspected_abort"
    assert verdicts[a3.row_id.hex] == "planned"


def test_short_first_attempt_is_NOT_abort_only_complete_redos_count():
    """If the earlier attempt is a 1-file derivative, it's trivial — not an abort.

    Only abort-flag when both members look like complete acquisitions
    (≥ ABORT_MIN_FILES files).
    """
    derivative = _row(n_files=1, acq_time="120000", image_type="M")
    actual = _row(n_files=200, acq_time="120030", image_type="M")
    chosen = {derivative.row_id.hex: _cls(derivative), actual.row_id.hex: _cls(actual)}
    verdicts = _detect_aborts([derivative, actual], chosen)
    assert verdicts[derivative.row_id.hex] == "trivial"
    assert verdicts[actual.row_id.hex] != "suspected_abort"

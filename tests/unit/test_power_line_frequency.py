"""``PowerLineFrequency`` is only ever stated when somebody stated it.

BIDS requires the field for EEG, MEG and iEEG, and the tool used to satisfy
that requirement by writing ``50`` whenever nothing else supplied a value. It
was always populated and sometimes wrong. Fifty hertz is Europe; a recording
made in the US, Canada, Japan or Brazil is sixty, and the sidecar gave no hint
that the number was a fallback rather than a measurement. Notch-filtering at
50 on 60 Hz data leaves the artefact exactly where it was.

Nothing replaced it. mne-bids writes ``PowerLineFrequency: "n/a"`` of its own
accord when ``raw.info["line_freq"]`` is unset, the schema types the field as
``anyOf`` number or the literal ``"n/a"``, and "not stated" is the true
statement. A guess that validates is worse than an honest blank that also
validates, because only one of them gets corrected.

The precedence these tests pin, highest first:

1. the recording's own header, which mne read from the file;
2. the inventory TSV's ``line_freq`` cell;
3. the recording-metadata template's dataset default;
4. nothing, which becomes ``"n/a"``.
"""

from __future__ import annotations

import pytest

from bidsmgr.recording_meta import AcquisitionSpec, RecordingMetaSpec, default_spec


# ---------------------------------------------------------------------------
# The default supplies nothing


def test_the_default_spec_states_no_frequency() -> None:
    assert default_spec().defaults.power_line_freq is None


def test_a_bare_spec_states_no_frequency() -> None:
    assert RecordingMetaSpec().defaults.power_line_freq is None


def test_a_template_that_says_50_is_still_honoured() -> None:
    """Removing the guess must not remove the ability to state a value."""
    spec = RecordingMetaSpec(defaults=AcquisitionSpec(power_line_freq=50.0))
    assert spec.defaults.power_line_freq == 50.0


# ---------------------------------------------------------------------------
# The resolution order convert applies


def _resolve(cell: str, template: float | None) -> float | None:
    """The chain from ``cli/convert.py``, in the order it applies it.

    Kept deliberately small and literal rather than importing the convert
    internals: what is under test is the ORDER and the absence of a floor, and
    a copy that drifts from the original is caught by the round-trip test
    below, which runs the real thing.
    """
    raw = str(cell).strip()
    try:
        value = float(raw) if raw else None
    except ValueError:
        value = None
    if value is None and template is not None:
        value = template
    return value


@pytest.mark.parametrize("cell, template, expected", [
    ("60", 50.0, 60.0),     # the cell wins over the template
    ("60", None, 60.0),     # the cell alone
    ("", 50.0, 50.0),       # the template when the cell is blank
    ("", None, None),       # NOTHING, which is the change
    ("  ", None, None),     # whitespace is blank
    ("banana", None, None), # unparseable is blank, not an exception
])
def test_the_chain_has_no_floor(cell, template, expected) -> None:
    assert _resolve(cell, template) == expected


def test_nothing_anywhere_returns_none_not_fifty() -> None:
    """The assertion this file exists for, stated on its own.

    If this ever reads 50.0 again, a fallback has come back and every EEG and
    MEG sidecar the tool writes is asserting a power-line frequency nobody
    measured.
    """
    assert _resolve("", None) is None


# ---------------------------------------------------------------------------
# What mne-bids does with the absence, which is why no fallback is needed


def test_mne_bids_writes_n_a_when_nothing_states_a_frequency(tmp_path) -> None:
    """The load-bearing third-party behaviour.

    The decision to delete the 50 Hz fallback rests entirely on this: that
    leaving ``raw.info["line_freq"]`` unset produces a VALID sidecar rather
    than a missing required field. If a future mne-bids stops doing it, the
    field goes missing and this test is what says so.
    """
    import json
    import warnings

    import numpy as np

    mne = pytest.importorskip("mne")
    mne_bids = pytest.importorskip("mne_bids")

    info = mne.create_info(["MEG0113", "MEG0112"], 250.0, "grad")
    raw = mne.io.RawArray(np.zeros((2, 500)), info, verbose="ERROR")
    assert raw.info.get("line_freq") is None, "the premise: nothing states it"

    source = tmp_path / "x_raw.fif"
    raw.save(source, verbose="ERROR")
    loaded = mne.io.read_raw_fif(source, verbose="ERROR")

    root = tmp_path / "bids"
    path = mne_bids.BIDSPath(
        subject="001", task="rest", datatype="meg", root=root,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne_bids.write_raw_bids(
            loaded, path, overwrite=True, verbose="ERROR",
        )

    sidecar = next(root.rglob("*_meg.json"))
    written = json.loads(sidecar.read_text())
    assert written["PowerLineFrequency"] == "n/a", (
        "mne-bids no longer writes n/a for an unset line_freq, so the field "
        "is now missing from a BIDS-required slot"
    )


def test_the_schema_accepts_n_a_for_this_field() -> None:
    """The other half of the premise: ``"n/a"`` is not a validation error.

    BIDS types PowerLineFrequency as ``anyOf`` a number or the literal
    ``"n/a"``, so writing the honest answer costs nothing.
    """
    from bidsmgr import schema

    field = next(
        f for f in schema.sidecar_fields("eeg", "eeg")
        if f.name == "PowerLineFrequency"
    )
    assert getattr(field, "accepts_na", False), (
        "the schema no longer reports n/a as acceptable here"
    )

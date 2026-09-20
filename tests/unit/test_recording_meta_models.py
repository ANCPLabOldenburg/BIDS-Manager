"""Unit tests for the recording_meta enrichment models + resolution."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bidsmgr.recording_meta import (
    AcquisitionSpec,
    AuxChannelSpec,
    FilterSpec,
    RecordingMetaSpec,
    bids_channel_types,
    default_spec,
    dump_spec,
    load_spec,
    merge_acquisition,
    resolve_effective,
)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_empty_spec_is_valid_and_additive():
    spec = RecordingMetaSpec()
    assert spec.schema_version == 1
    assert spec.defaults.power_line_freq is None
    assert spec.overrides == {}


def test_default_spec_invents_no_power_line_frequency():
    """It used to return 50 Hz, and that was a guess dressed as a measurement.

    Fifty is Europe. A recording made in the US, Canada, Japan or Brazil is
    sixty, and the old default reached the sidecar of every recording whose
    header did not state one, in a BIDS-REQUIRED field, indistinguishable from
    a number somebody had actually checked.

    Nothing replaces it: mne-bids writes ``PowerLineFrequency: "n/a"`` when
    ``raw.info["line_freq"]`` is unset, which the schema permits and which is
    true.
    """
    assert default_spec().defaults.power_line_freq is None


def test_a_caller_that_knows_can_still_say_so():
    """The parameter stays; only the default changed."""
    assert default_spec(60.0).defaults.power_line_freq == 60.0


def test_common_manufacturers_exported_pure_data():
    from bidsmgr.recording_meta import COMMON_CAP_MANUFACTURERS, COMMON_MANUFACTURERS
    assert isinstance(COMMON_MANUFACTURERS, tuple)
    assert "Brain Products" in COMMON_MANUFACTURERS
    assert "MEGIN / Elekta / Neuromag" in COMMON_MANUFACTURERS
    assert "CTF" in COMMON_MANUFACTURERS              # MEG
    assert "NIRx" in COMMON_MANUFACTURERS             # fNIRS
    assert "EasyCap" in COMMON_CAP_MANUFACTURERS


def test_meg_fields_on_acquisition_spec():
    # Only the manual MEG fields exist; channel-derived ones are mne-bids's job.
    acq = AcquisitionSpec(
        dewar_position="supine",
        associated_empty_room="bids::sub-x",
        subject_artefact_description="movement",
    )
    assert acq.dewar_position == "supine"
    assert acq.associated_empty_room == "bids::sub-x"
    # The auto-derived MEG fields are NOT part of the model.
    assert not hasattr(acq, "continuous_head_localization")
    assert not hasattr(acq, "digitized_landmarks")
    # Additive: round-trips through JSON by attr name.
    dumped = dump_spec(RecordingMetaSpec(defaults=acq))
    assert "dewar_position" in dumped


def test_aux_channel_bids_type_validated_against_schema():
    # A real BIDS channel type passes.
    ok = AuxChannelSpec(mne_type="ecg", bids_type="ECG", units="mV")
    assert ok.bids_type == "ECG"
    # The schema vocabulary actually contains it.
    assert "ECG" in bids_channel_types()


def test_aux_channel_rejects_unknown_bids_type():
    with pytest.raises(ValidationError):
        AuxChannelSpec(bids_type="NOTATYPE")


def test_aux_channel_rejects_unknown_mne_type():
    with pytest.raises(ValidationError):
        AuxChannelSpec(mne_type="not_a_real_mne_type")


def test_filter_kind_constrained():
    FilterSpec(name="lp", kind="Hardware", info={"cutoff": 260})
    with pytest.raises(ValidationError):
        FilterSpec(name="lp", kind="firmware", info={})


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        AcquisitionSpec(not_a_field="x")


# ---------------------------------------------------------------------------
# Resolution / precedence
# ---------------------------------------------------------------------------


def test_override_wins_over_default_scalar():
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(eeg_reference="Cz", eeg_ground="AFz"),
        overrides={"row-1": AcquisitionSpec(eeg_reference="FCz")},
    )
    eff = resolve_effective(spec, "row-1")
    assert eff.acquisition.eeg_reference == "FCz"  # override wins
    assert eff.acquisition.eeg_ground == "AFz"  # default retained


def test_unset_override_field_keeps_default():
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(manufacturer="Brain Products"),
        overrides={"row-1": AcquisitionSpec(eeg_reference="Cz")},
    )
    eff = resolve_effective(spec, "row-1")
    assert eff.acquisition.manufacturer == "Brain Products"
    assert eff.acquisition.eeg_reference == "Cz"


def test_aux_channels_merge_by_key():
    base = AcquisitionSpec(aux_channels={"ECG": AuxChannelSpec(bids_type="ECG")})
    over = AcquisitionSpec(aux_channels={"EOG": AuxChannelSpec(bids_type="EOG")})
    merged = merge_acquisition(base, over)
    assert set(merged.aux_channels) == {"ECG", "EOG"}


def test_filters_replace_when_override_nonempty():
    base = AcquisitionSpec(filters=[FilterSpec(name="a", kind="Hardware")])
    over = AcquisitionSpec(filters=[FilterSpec(name="b", kind="Software")])
    merged = merge_acquisition(base, over)
    assert [f.name for f in merged.filters] == ["b"]


def test_event_map_task_then_global_fallback():
    spec = RecordingMetaSpec(
        event_maps={"rest": {"S 20": "eyes_open"}, "*": {"S 99": "marker"}},
    )
    assert resolve_effective(spec, "row", "rest").event_map == {"S 20": "eyes_open"}
    assert resolve_effective(spec, "row", "other").event_map == {"S 99": "marker"}
    assert resolve_effective(spec, "row", None).event_map == {"S 99": "marker"}


def test_missing_row_resolves_to_defaults_only():
    spec = RecordingMetaSpec(defaults=AcquisitionSpec(eeg_reference="Cz"))
    eff = resolve_effective(spec, "no-such-row")
    assert eff.acquisition.eeg_reference == "Cz"


# ---------------------------------------------------------------------------
# Round-trip I/O
# ---------------------------------------------------------------------------


def test_round_trip_json(tmp_path):
    spec = RecordingMetaSpec(
        defaults=AcquisitionSpec(
            eeg_reference="Cz",
            aux_channels={"ECG": AuxChannelSpec(mne_type="ecg", bids_type="ECG")},
            filters=[FilterSpec(name="lp", kind="Software", info={"cutoff": 260})],
        ),
        event_maps={"rest": {"S 20": "eyes_open"}},
    )
    path = tmp_path / "meta.json"
    path.write_text(dump_spec(spec), encoding="utf-8")
    reloaded = load_spec(path)
    assert reloaded.defaults.eeg_reference == "Cz"
    assert reloaded.defaults.aux_channels["ECG"].bids_type == "ECG"
    assert reloaded.event_maps["rest"]["S 20"] == "eyes_open"


def test_load_rejects_unknown_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"defaults": {"bogus": 1}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_spec(path)

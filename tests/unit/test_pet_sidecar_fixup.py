"""The PET metadata spec and the sidecar fixup that applies it.

The fixup does four things in a fixed order: rename DICOM-derived keys to their
BIDS names, fill from the user's spec, repair fields the schema types as arrays,
and prune what does not belong in a shareable sidecar. Order is load-bearing,
so it is asserted rather than assumed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from bidsmgr.fixups.pet_sidecar import (
    IDENTIFYING_KEYS,
    NON_BIDS_KEYS,
    _array_typed_keys,
    enrich_pet_sidecars,
)
from bidsmgr.recording_meta import (
    PetAcquisitionSpec,
    RecordingMetaSpec,
    merge_pet,
    resolve_pet,
)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def test_pet_block_inherits_dataset_defaults() -> None:
    spec = RecordingMetaSpec(
        pet_defaults=PetAcquisitionSpec(tracer_name="FDG", units="Bq/mL"),
        pet_overrides={"row9": PetAcquisitionSpec(injected_radioactivity=44.4)},
    )
    inherited = resolve_pet(spec, "anything-else")
    assert inherited.tracer_name == "FDG"
    assert inherited.injected_radioactivity is None

    overridden = resolve_pet(spec, "row9")
    assert overridden.tracer_name == "FDG", "unset override fields keep the default"
    assert overridden.injected_radioactivity == 44.4


def test_override_wins_over_the_default() -> None:
    merged = merge_pet(
        PetAcquisitionSpec(tracer_name="FDG"),
        PetAcquisitionSpec(tracer_name="PIB"),
    )
    assert merged.tracer_name == "PIB"


def test_recon_parameter_lists_are_replaced_as_a_unit() -> None:
    """The three lists are positional, so interleaving would mis-pair them."""
    merged = merge_pet(
        PetAcquisitionSpec(
            recon_method_parameter_labels=["iterations", "subsets"],
            recon_method_parameter_values=[3, 21],
        ),
        PetAcquisitionSpec(
            recon_method_parameter_labels=["lambda"],
            recon_method_parameter_values=[0.5],
        ),
    )
    assert merged.recon_method_parameter_labels == ["lambda"]
    assert merged.recon_method_parameter_values == [0.5]


def test_a_scaffold_written_before_pet_support_still_loads() -> None:
    """REGRESSION: the PET block is additive; released scaffolds must survive."""
    old = json.dumps({
        "schema_version": 1,
        "defaults": {"power_line_freq": 50.0, "montage": "standard_1020"},
        "event_maps": {"*": {"S 1": "go"}},
    })
    spec = RecordingMetaSpec.model_validate_json(old)
    assert spec.defaults.power_line_freq == 50.0
    assert spec.event_maps == {"*": {"S 1": "go"}}
    assert spec.pet_defaults.tracer_name is None
    assert spec.pet_overrides == {}


# ---------------------------------------------------------------------------
# The fixup
# ---------------------------------------------------------------------------


def _staged(tmp_path, sidecar: dict, basename: str = "sub-001_pet"):
    """Write a staged PET sidecar and return (staging_dir, tasks, path)."""
    pet_dir = tmp_path / "sub-001" / "pet"
    pet_dir.mkdir(parents=True)
    path = pet_dir / f"{basename}.json"
    path.write_text(json.dumps(sidecar))
    task = SimpleNamespace(datatype="pet", basename=basename, row_id="row9")
    return tmp_path / "sub-001", [task], path


def test_dicom_key_is_renamed_to_its_bids_name(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {"ReconstructionMethod": "PSF+TOF"})
    assert enrich_pet_sidecars(staging, tasks, None) == 1
    out = json.loads(path.read_text())
    assert out["ReconMethodName"] == "PSF+TOF"
    assert "ReconstructionMethod" not in out


def test_rename_never_clobbers_an_existing_bids_key(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {
        "ReconstructionMethod": "from DICOM",
        "ReconMethodName": "already correct",
    })
    enrich_pet_sidecars(staging, tasks, None)
    out = json.loads(path.read_text())
    assert out["ReconMethodName"] == "already correct"


def test_spec_fills_the_fields_no_scanner_records(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {"Units": "Bq/mL"})
    spec = RecordingMetaSpec(pet_overrides={"row9": PetAcquisitionSpec(
        tracer_name="FDG",
        tracer_radionuclide="F18",
        injected_radioactivity=44.4,
        injected_radioactivity_units="MBq",
        mode_of_administration="bolus",
    )})
    assert enrich_pet_sidecars(staging, tasks, spec) == 1
    out = json.loads(path.read_text())
    assert out["TracerName"] == "FDG"
    assert out["InjectedRadioactivity"] == 44.4
    assert out["ModeOfAdministration"] == "bolus"
    assert out["Units"] == "Bq/mL", "untouched converter values survive"


def test_a_user_value_beats_the_converter(tmp_path) -> None:
    """A spec entry is a statement; the converter's is an inference."""
    staging, tasks, path = _staged(tmp_path, {"Units": "counts"})
    spec = RecordingMetaSpec(
        pet_defaults=PetAcquisitionSpec(units="Bq/mL"),
    )
    enrich_pet_sidecars(staging, tasks, spec)
    assert json.loads(path.read_text())["Units"] == "Bq/mL"


def test_an_unset_field_leaves_the_converter_value_alone(tmp_path) -> None:
    """Enrichment is additive: a blank must not erase a good value."""
    staging, tasks, path = _staged(tmp_path, {"Units": "Bq/mL"})
    spec = RecordingMetaSpec(pet_defaults=PetAcquisitionSpec(tracer_name="FDG"))
    enrich_pet_sidecars(staging, tasks, spec)
    assert json.loads(path.read_text())["Units"] == "Bq/mL"


def test_scalar_is_wrapped_for_fields_the_schema_types_as_arrays(tmp_path) -> None:
    """dcm2niix writes a bare scalar for a single frame; BIDS wants an array."""
    staging, tasks, path = _staged(tmp_path, {
        "ScatterFraction": 0.295,
        "FrameDuration": 300,
        "DecayCorrectionFactor": 1.47,
    })
    enrich_pet_sidecars(staging, tasks, None)
    out = json.loads(path.read_text())
    assert out["ScatterFraction"] == [0.295]
    assert out["FrameDuration"] == [300]
    assert out["DecayCorrectionFactor"] == [1.47]


def test_scalar_fields_are_not_wrapped(tmp_path) -> None:
    """The schema types DoseCalibrationFactor as a number, so leave it."""
    staging, tasks, path = _staged(tmp_path, {"DoseCalibrationFactor": 32156500})
    enrich_pet_sidecars(staging, tasks, None)
    assert json.loads(path.read_text())["DoseCalibrationFactor"] == 32156500


def test_array_typed_keys_come_from_the_schema() -> None:
    keys = _array_typed_keys()
    assert "FrameDuration" in keys
    assert "ScatterFraction" in keys
    # A number in the schema, and an anyOf, must both stay out.
    assert "DoseCalibrationFactor" not in keys
    assert "ReconFilterSize" not in keys


def test_existing_arrays_are_untouched(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {"FrameDuration": [300, 300, 600]})
    enrich_pet_sidecars(staging, tasks, None)
    assert json.loads(path.read_text())["FrameDuration"] == [300, 300, 600]


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_patient_identifiers_are_pruned(tmp_path) -> None:
    """dcm2niix runs with -ba n to keep the UIDs, which keeps these too."""
    staging, tasks, path = _staged(tmp_path, {
        "PatientName": "DOE^JANE",
        "PatientID": "12345",
        "PatientBirthDate": "1970-01-01",
        "AccessionNumber": "A1",
        "ReferringPhysicianName": "SMITH",
        "Units": "Bq/mL",
    })
    enrich_pet_sidecars(staging, tasks, None)
    out = json.loads(path.read_text())
    assert not (IDENTIFYING_KEYS & set(out))
    assert out["Units"] == "Bq/mL"


def test_uids_are_kept_for_provenance(tmp_path) -> None:
    """Pseudonymous and load-bearing: they trace an image to its source."""
    staging, tasks, path = _staged(tmp_path, {
        "SeriesInstanceUID": "1.2.3", "StudyInstanceUID": "4.5.6",
        "PatientName": "DOE^JANE",
    })
    enrich_pet_sidecars(staging, tasks, None)
    out = json.loads(path.read_text())
    assert out["SeriesInstanceUID"] == "1.2.3"
    assert out["StudyInstanceUID"] == "4.5.6"
    assert "PatientName" not in out


def test_pruning_can_be_turned_off(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {"PatientName": "DOE^JANE"})
    enrich_pet_sidecars(staging, tasks, None, prune_identifiers=False)
    assert json.loads(path.read_text())["PatientName"] == "DOE^JANE"


def test_non_bids_noise_is_pruned_regardless(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {
        "BidsGuess": ["PET", "PET"], "Modality": "PT", "Units": "Bq/mL",
    })
    enrich_pet_sidecars(staging, tasks, None, prune_identifiers=False)
    out = json.loads(path.read_text())
    assert not (NON_BIDS_KEYS & set(out))
    assert out["Units"] == "Bq/mL"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_non_pet_tasks_are_ignored(tmp_path) -> None:
    """REGRESSION: an MRI or EEG sidecar must not be touched by this fixup."""
    pet_dir = tmp_path / "sub-001" / "anat"
    pet_dir.mkdir(parents=True)
    path = pet_dir / "sub-001_T1w.json"
    path.write_text(json.dumps({"PatientName": "DOE^JANE", "Modality": "MR"}))
    task = SimpleNamespace(datatype="anat", basename="sub-001_T1w", row_id="r")

    assert enrich_pet_sidecars(tmp_path / "sub-001", [task], None) == 0
    assert json.loads(path.read_text())["PatientName"] == "DOE^JANE"


def test_a_clean_sidecar_is_left_alone(tmp_path) -> None:
    """No change means no rewrite, so mtimes stay meaningful."""
    staging, tasks, path = _staged(tmp_path, {"Units": "Bq/mL", "TracerName": "FDG"})
    assert enrich_pet_sidecars(staging, tasks, None) == 0


def test_a_missing_sidecar_is_not_an_error(tmp_path) -> None:
    (tmp_path / "sub-001").mkdir()
    task = SimpleNamespace(datatype="pet", basename="sub-001_pet", row_id="r")
    assert enrich_pet_sidecars(tmp_path / "sub-001", [task], None) == 0


def test_unreadable_json_is_survived(tmp_path) -> None:
    staging, tasks, path = _staged(tmp_path, {})
    path.write_text("{ not json")
    assert enrich_pet_sidecars(staging, tasks, None) == 0

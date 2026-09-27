"""Reading a PET metadata JSON: the third way an operator can supply a dose.

Not a new format. It is the shape pypet2bids accepts through
``--set-default-metadata-json``, so a lab that already has one of those files
hands us the same file.
"""

from __future__ import annotations

import json

from bidsmgr.metadata.pet_metadata_json import read_pet_metadata_json


def _write(tmp_path, payload):
    path = tmp_path / "dose.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestTheShapes:
    def test_a_flat_object_is_the_dataset_default(self, tmp_path):
        path = _write(tmp_path, {
            "TracerName": "FDG",
            "InjectedRadioactivity": 81.24,
            "InjectedRadioactivityUnits": "MBq",
        })
        blocks = read_pet_metadata_json(path)
        assert list(blocks) == [""]
        assert blocks[""].tracer_name == "FDG"
        assert blocks[""].injected_radioactivity == 81.24

    def test_subject_keys_scope_the_blocks(self, tmp_path):
        path = _write(tmp_path, {
            "sub-01": {"InjectedRadioactivity": 81.24},
            "sub-02": {"InjectedRadioactivity": 74.10},
        })
        blocks = read_pet_metadata_json(path)
        assert sorted(blocks) == ["sub-01", "sub-02"]
        assert blocks["sub-02"].injected_radioactivity == 74.10

    def test_scoped_and_unscoped_can_coexist(self, tmp_path):
        """The unscoped keys are the default the scoped ones sit on top of."""
        path = _write(tmp_path, {
            "TracerName": "FDG",
            "sub-01": {"InjectedRadioactivity": 81.24},
        })
        blocks = read_pet_metadata_json(path)
        assert blocks[""].tracer_name == "FDG"
        assert blocks["sub-01"].injected_radioactivity == 81.24

    def test_the_pet2bids_wrapper_is_unwrapped(self, tmp_path):
        """Their file works unmodified, blood blocks and all."""
        path = _write(tmp_path, {
            "nifti_json": {"TracerName": "FDG"},
            "blood_json": {"PlasmaAvail": True},
            "blood_tsv": {},
        })
        blocks = read_pet_metadata_json(path)
        assert blocks[""].tracer_name == "FDG"


class TestWhatItRefuses:
    def test_an_unknown_key_is_dropped_not_written(self, tmp_path, caplog):
        """An unrestricted overlay is a way to put arbitrary JSON in a dataset."""
        path = _write(tmp_path, {"TracerName": "FDG", "NotABidsField": "x"})
        blocks = read_pet_metadata_json(path)
        assert blocks[""].tracer_name == "FDG"
        assert "NotABidsField" not in blocks[""].model_dump()
        assert "NotABidsField" in caplog.text

    def test_a_missing_file_yields_nothing(self, tmp_path):
        """A metadata import must never abort a conversion."""
        assert read_pet_metadata_json(tmp_path / "absent.json") == {}

    def test_broken_json_yields_nothing(self, tmp_path):
        path = tmp_path / "dose.json"
        path.write_text("{not json", encoding="utf-8")
        assert read_pet_metadata_json(path) == {}

    def test_a_json_array_yields_nothing(self, tmp_path):
        assert read_pet_metadata_json(_write(tmp_path, [1, 2, 3])) == {}

    def test_blank_values_are_skipped(self, tmp_path):
        path = _write(tmp_path, {"TracerName": "", "InjectedMass": None,
                                 "TracerRadionuclide": "F18"})
        blocks = read_pet_metadata_json(path)
        assert blocks[""].tracer_radionuclide == "F18"
        assert blocks[""].tracer_name is None


def test_every_field_the_tool_can_write_can_also_be_read(tmp_path):
    """The reader is inverted from the maps that WRITE these fields.

    So a field added to the spec becomes readable without anyone having to
    remember to add it in a second place. This is the test that notices if
    that inversion is ever replaced by a hand-maintained list.
    """
    from bidsmgr.metadata.pet_metadata_json import _bids_to_field
    from bidsmgr.recording_meta.chain import PET_LIST_TO_BIDS, PET_SCALAR_TO_BIDS

    writable = set(PET_SCALAR_TO_BIDS.values()) | set(PET_LIST_TO_BIDS.values())
    assert set(_bids_to_field()) == writable


class TestTheShapesRealFilesCarry:
    """A file written for pet2bids, or copied out of a finished sidecar."""

    def test_a_one_element_array_where_the_spec_wants_a_scalar(self, tmp_path):
        """BIDS types ReconFilterSize as an array; the spec models a scalar,
        because a value is stated once and the conversion wraps it. Refusing
        the file over a bracket would be refusing the shape the standard
        describes."""
        path = _write(tmp_path, {"ReconFilterSize": [0], "TracerName": "FDG"})
        blocks = read_pet_metadata_json(path)
        assert blocks[""].recon_filter_size == 0
        assert blocks[""].tracer_name == "FDG"

    def test_a_genuine_list_field_keeps_its_list(self, tmp_path):
        path = _write(tmp_path, {"ReconMethodParameterLabels": ["subsets"]})
        blocks = read_pet_metadata_json(path)
        assert blocks[""].recon_method_parameter_labels == ["subsets"]

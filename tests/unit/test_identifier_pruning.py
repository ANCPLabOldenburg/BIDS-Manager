"""Identifiers must not reach a BIDS dataset, from a sidecar or from inside
an image.

BIDS Manager runs dcm2niix with ``-ba n`` so ``SeriesInstanceUID`` survives
for provenance, and that flag keeps the patient identifiers alongside it.
Until these passes existed, only the PET fixup pruned them: measured on this
lab's own data, 58 of 71 MRI sidecars carried the participant's name, ID,
date of birth, age, sex, size and weight, plus the institution's street
address. MR spectroscopy was worse, because its metadata lives INSIDE the
.nii.gz where no sidecar pass can reach.
"""

from __future__ import annotations

import json

import pytest

from bidsmgr.fixups.identifiers import (
    IDENTIFYING_KEYS,
    prune_identifiers,
    strip_identifiers,
)

IDENTIFYING = {
    "PatientName": "OL_3846^YY11YY11",
    "PatientID": "OL_3846",
    "PatientBirthDate": "1993-01-01",
    "PatientSex": "M",
    "PatientAge": "032Y",
    "PatientWeight": 66,
    "PatientSize": 1.8,
    "AccessionNumber": "12345",
    "OperatorsName": "TECH",
}

KEPT = {
    # A BIDS RECOMMENDED field, about the site and not the participant.
    # Pruning it would be this tool overriding the standard on a privacy
    # question the standard has already answered.
    "InstitutionAddress": "Kuepkersweg 74,Oldenburg,DE",
    "InstitutionName": "Uni Oldenburg",
    # Pseudonymous, and what lets an image be traced to its source series.
    "SeriesInstanceUID": "1.2.3.4",
    "StudyInstanceUID": "1.2.3",
    # Geometry, not identity: where the person lay, not who they are.
    "PatientPosition": "HFS",
    "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
    # Ordinary metadata.
    "Manufacturer": "Siemens",
    "RepetitionTime": 2.0,
}


class TestWhatCountsAsIdentifying:
    @pytest.mark.parametrize("key", sorted(IDENTIFYING))
    def test_it_goes(self, key):
        data = {key: IDENTIFYING[key]}
        assert strip_identifiers(data) == [key]
        assert data == {}

    @pytest.mark.parametrize("key", sorted(KEPT))
    def test_what_stays_stays(self, key):
        data = {key: KEPT[key]}
        assert strip_identifiers(data) == []
        assert data == {key: KEPT[key]}

    def test_both_spellings_of_a_birth_date_are_known(self):
        """DICOM says PatientBirthDate; NIfTI-MRS says PatientDoB. Same fact,
        and a key one pass removes while another leaves is worse than a key
        neither removes: it makes the dataset look cleaned."""
        assert "PatientBirthDate" in IDENTIFYING_KEYS
        assert "PatientDoB" in IDENTIFYING_KEYS

    def test_source_paths_go_by_default(self):
        """Not identifying in itself, but it carries a local directory
        layout into a dataset meant to be shareable."""
        data = {"OriginalFile": ["/home/someone/scans/000.dcm"]}
        assert strip_identifiers(data) == ["OriginalFile"]

    def test_source_paths_can_be_kept_when_asked(self):
        data = {"OriginalFile": ["a.dcm"]}
        assert strip_identifiers(data, paths=False) == []


class TestPruningAStagedTree:
    def test_every_sidecar_is_cleaned_whatever_the_datatype(self, tmp_path):
        """This is the point. It used to be PET only, so a PET dataset was
        clean and an MRI one was not."""
        made = []
        for datatype in ("anat", "func", "dwi", "fmap", "mrs", "pet"):
            folder = tmp_path / "sub-01" / datatype
            folder.mkdir(parents=True)
            p = folder / f"sub-01_{datatype}.json"
            p.write_text(json.dumps({**IDENTIFYING, **KEPT}))
            made.append(p)

        assert prune_identifiers(tmp_path) == len(made)
        for p in made:
            data = json.loads(p.read_text())
            assert not (set(data) & IDENTIFYING_KEYS), p.name
            assert data["SeriesInstanceUID"] == "1.2.3.4"
            assert data["PatientPosition"] == "HFS"

    def test_a_clean_sidecar_is_not_rewritten(self, tmp_path):
        """A conversion that rewrites every file it touches makes its own
        logs useless for telling which files it changed."""
        folder = tmp_path / "sub-01" / "anat"
        folder.mkdir(parents=True)
        p = folder / "sub-01_T1w.json"
        p.write_text(json.dumps(KEPT))
        before = p.read_text()
        assert prune_identifiers(tmp_path) == 0
        assert p.read_text() == before

    def test_our_own_bookkeeping_is_left_alone(self, tmp_path):
        hidden = tmp_path / ".bidsmgr" / "project"
        hidden.mkdir(parents=True)
        p = hidden / "state.json"
        p.write_text(json.dumps({"PatientName": "kept: not a sidecar"}))
        assert prune_identifiers(tmp_path) == 0
        assert "PatientName" in p.read_text()

    def test_a_malformed_json_does_not_abort_the_pass(self, tmp_path):
        folder = tmp_path / "sub-01" / "anat"
        folder.mkdir(parents=True)
        (folder / "broken.json").write_text("{not json")
        good = folder / "sub-01_T1w.json"
        good.write_text(json.dumps(IDENTIFYING))
        assert prune_identifiers(tmp_path) == 1
        assert json.loads(good.read_text()) == {}

"""A dose sheet written one row per FIELD rather than one row per scan.

The natural way to write a study with a single dose, and what BIDS Manager's
own PET sample ships. It silently did nothing until the reader learned the
shape: the sample's documented "with the sheet, no errors" was not
reproducible, because the reader looked for a participant column, did not
find one, and ignored the file.
"""

from __future__ import annotations

import pytest

from bidsmgr.metadata.pet_spreadsheet import read_pet_spreadsheet


def _sheet(tmp_path, text: str):
    path = tmp_path / "dose.csv"
    path.write_text(text, encoding="utf-8")
    return path


class TestTheVerticalShape:
    def test_field_and_value_columns_are_read(self, tmp_path):
        path = _sheet(tmp_path, (
            "field,value,units,what it is\n"
            "TracerName,FDG,,The radiolabelled compound\n"
            "InjectedRadioactivity,81.24,MBq,Total activity injected\n"
        ))
        blocks = read_pet_spreadsheet(path)
        assert list(blocks) == [""]        # applies to every PET run
        assert blocks[""].tracer_name == "FDG"
        assert blocks[""].injected_radioactivity == 81.24

    def test_the_empty_key_means_every_run(self, tmp_path):
        """Same convention as the JSON route, so the two cannot disagree
        about what a study-wide dose means."""
        path = _sheet(tmp_path, "field,value\nTracerName,FDG\n")
        assert list(read_pet_spreadsheet(path)) == [""]

    def test_an_array_field_is_split_from_one_cell(self, tmp_path):
        """BIDS types the reconstruction-parameter trio as arrays, and a
        spreadsheet cell is one string."""
        path = _sheet(tmp_path, (
            "field,value\n"
            "ReconMethodParameterLabels,subsets;iterations\n"
            "ReconMethodParameterValues,21;3\n"
        ))
        block = read_pet_spreadsheet(path)[""]
        assert block.recon_method_parameter_labels == ["subsets", "iterations"]
        assert block.recon_method_parameter_values == [21, 3]

    def test_a_row_naming_no_pet_field_is_reported_not_written(
        self, tmp_path, caplog,
    ):
        path = _sheet(tmp_path, "field,value\nTracerName,FDG\nNonsense,x\n")
        block = read_pet_spreadsheet(path)[""]
        assert block.tracer_name == "FDG"
        assert "Nonsense" in caplog.text

    def test_the_row_per_scan_shape_still_wins_when_both_could_match(
        self, tmp_path,
    ):
        """A sheet with an identifying column is the per-scan shape, even if
        it happens to have a column called value."""
        path = _sheet(tmp_path, (
            "participant_id,TracerName,value\n"
            "sub-01,FDG,ignored\n"
        ))
        blocks = read_pet_spreadsheet(path)
        assert list(blocks) == ["sub-01"]

    def test_a_sheet_that_is_neither_shape_is_ignored(self, tmp_path, caplog):
        path = _sheet(tmp_path, "a,b\n1,2\n")
        assert read_pet_spreadsheet(path) == {}
        assert "ignoring the file" in caplog.text


@pytest.mark.parametrize("header", ["field,value", "key,value", "name,answer"])
def test_the_column_names_people_actually_use(tmp_path, header):
    path = tmp_path / "d.csv"
    path.write_text(f"{header}\nTracerName,FDG\n", encoding="utf-8")
    assert read_pet_spreadsheet(path)[""].tracer_name == "FDG"

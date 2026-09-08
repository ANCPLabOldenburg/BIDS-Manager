"""Blood curves become correctly named, correctly described BIDS tables.

Two layers are tested and they are deliberately different in what they need.

The contract of our wrapper, which is the part we wrote, is tested with a stub:
the tags we accept, the names we build, and the promise that a blood problem
never costs a user their imaging conversion. That runs anywhere.

The conversion itself needs real PMOD exports and the real dependency, so those
tests are gated on ``BIDS_MANAGER_REAL_PET_BLOOD=1`` beside the other real-data
gates. The files came from the pet2bids repository; see ``PROVENANCE.md`` beside
them.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidsmgr.fixups.blood import (
    BLOOD_SERIES,
    blood_role,
    convert_blood_files,
    is_blood_role,
    parse_blood_role,
)

BLOOD_DATA = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/PET_BLOOD/pmod"
)
real_blood = pytest.mark.skipif(
    os.environ.get("BIDS_MANAGER_REAL_PET_BLOOD") != "1" or not BLOOD_DATA.is_dir(),
    reason="needs BIDS_MANAGER_REAL_PET_BLOOD=1 and the PMOD example files",
)


def _task(basename: str, companions):
    return SimpleNamespace(basename=basename, companion_files=tuple(companions))


def _staged(tmp_path: Path, basename: str) -> Path:
    """A staged subject with one PET run already converted."""
    staging = tmp_path / "sub-001"
    pet = staging / "pet"
    pet.mkdir(parents=True)
    (pet / f"{basename}.json").write_text("{}", encoding="utf-8")
    return staging


# --------------------------------------------------------------------------
# the tags
# --------------------------------------------------------------------------


@pytest.mark.parametrize("series", sorted(BLOOD_SERIES))
@pytest.mark.parametrize("method", ["manual", "automatic"])
def test_role_round_trips(series: str, method: str) -> None:
    assert parse_blood_role(blood_role(series, method)) == (series, method)


@pytest.mark.parametrize(
    "tag",
    [
        "events",                    # an ordinary companion
        "blood:",                    # nothing after the prefix
        "blood:plasma",              # no method
        "blood:plasma:manual:extra", # too many parts
        "blood:spinalfluid:manual",  # not a series we know
        "blood:plasma:guessed",      # not a method we know
    ],
)
def test_a_tag_we_do_not_own_parses_to_nothing(tag: str) -> None:
    """User data, so one bad entry returns None rather than raising."""
    assert parse_blood_role(tag) is None


def test_the_companion_copier_is_told_to_stand_back() -> None:
    """Blood is converted, not copied, so the copier must skip these."""
    assert is_blood_role(blood_role("plasma", "manual"))
    assert not is_blood_role("events")


# --------------------------------------------------------------------------
# never at the cost of the imaging run
# --------------------------------------------------------------------------


def test_a_run_with_no_blood_does_nothing(tmp_path: Path) -> None:
    staging = _staged(tmp_path, "sub-001_pet")
    assert convert_blood_files(staging, [_task("sub-001_pet", [])]) == 0


def test_a_missing_source_is_skipped_not_raised(tmp_path: Path) -> None:
    staging = _staged(tmp_path, "sub-001_pet")
    task = _task(
        "sub-001_pet",
        [(blood_role("plasma", "manual"), str(tmp_path / "gone.bld"))],
    )
    assert convert_blood_files(staging, [task]) == 0


def test_an_absent_dependency_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Without pypet2bids, blood says why and the conversion survives."""
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")
    monkeypatch.setattr("bidsmgr.fixups.blood._load_converter", lambda: None)

    staging = _staged(tmp_path, "sub-001_pet")
    task = _task("sub-001_pet", [(blood_role("plasma", "manual"), str(curve))])
    assert convert_blood_files(staging, [task]) == 0


def test_a_converter_that_throws_loses_only_the_blood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")

    def explode(**_kwargs):
        raise RuntimeError("unreadable curve")

    monkeypatch.setattr("bidsmgr.fixups.blood._load_converter", lambda: explode)
    staging = _staged(tmp_path, "sub-001_pet")
    task = _task("sub-001_pet", [(blood_role("plasma", "manual"), str(curve))])

    assert convert_blood_files(staging, [task]) == 0
    assert (staging / "pet" / "sub-001_pet.json").is_file()


def test_the_method_is_always_stated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard against a hung GUI.

    ``PmodToBlood`` calls ``input()`` when it cannot tell manual sampling from
    automatic. In a window with no console that is a freeze with no visible
    cause, so every series must carry its method.
    """
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("bidsmgr.fixups.blood._load_converter", lambda: record)

    curves = {}
    for series in BLOOD_SERIES:
        path = tmp_path / f"{series}.bld"
        path.write_text("time\tactivity\n", encoding="utf-8")
        curves[series] = path

    staging = _staged(tmp_path, "sub-001_pet")
    task = _task(
        "sub-001_pet",
        [(blood_role(s, "manual"), str(p)) for s, p in curves.items()],
    )
    convert_blood_files(staging, [task])

    for argument in BLOOD_SERIES.values():
        assert argument in seen
        assert seen[f"{argument}_collection_method"] == "manual"


def test_names_come_from_the_runs_own_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entities are the recording's, never string-built here.

    Left to itself pet2bids names output after the output directory, so two
    tracers in one subject would collide.
    """
    def produce(output_path, **_kwargs):
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "whatever_recording-manual_blood.tsv").write_text(
            "time\tplasma_radioactivity\n0\t1\n", encoding="utf-8"
        )
        (out / "whatever_blood.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("bidsmgr.fixups.blood._load_converter", lambda: produce)

    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")
    basename = "sub-001_ses-baseline_trc-PIB_run-2_pet"
    staging = _staged(tmp_path, basename)
    task = _task(basename, [(blood_role("plasma", "manual"), str(curve))])

    assert convert_blood_files(staging, [task]) == 2
    written = {p.name for p in (staging / "pet").iterdir()}
    assert "sub-001_ses-baseline_trc-PIB_run-2_recording-manual_blood.tsv" in written
    # ``recording`` is required on blood files, sidecar included. pet2bids
    # 1.5.1 omits it from the sidecar, so the entity is taken from the table
    # rather than copied through, which would pair a valid table with an
    # invalid sidecar.
    assert "sub-001_ses-baseline_trc-PIB_run-2_recording-manual_blood.json" in written


# --------------------------------------------------------------------------
# the data dictionary
# --------------------------------------------------------------------------


def test_the_dictionary_describes_the_table_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this rewrite exists for.

    pet2bids 1.5.1 describes a metabolite_parent_fraction column that is not in
    the table and omits the plasma_radioactivity column that is. A sidecar that
    describes columns a file does not have fails validation, so the dictionary
    is rebuilt from what was actually written.
    """
    def produce_a_mismatched_pair(output_path, **_kwargs):
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "x_recording-manual_blood.tsv").write_text(
            "time\tplasma_radioactivity\twhole_blood_radioactivity\n0\t1\t2\n",
            encoding="utf-8",
        )
        (out / "x_blood.json").write_text(
            json.dumps(
                {
                    "Time": {"Description": "a column that does not exist"},
                    "MetaboliteMethod": "HPLC",
                    "metabolite_parent_fraction": {"Description": "not in the table"},
                    "whole_blood_radioactivity": {
                        "Description": "theirs",
                        "Units": "kBq/mL",
                    },
                    "MetaboliteAvail": True,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "bidsmgr.fixups.blood._load_converter", lambda: produce_a_mismatched_pair
    )
    curve = tmp_path / "plasma.bld"
    curve.write_text("time\tactivity\n", encoding="utf-8")
    staging = _staged(tmp_path, "sub-001_pet")
    task = _task("sub-001_pet", [(blood_role("plasma", "manual"), str(curve))])
    convert_blood_files(staging, [task])

    written = json.loads(
        (staging / "pet" / "sub-001_recording-manual_blood.json")
        .read_text(encoding="utf-8")
    )
    described = {k for k, v in written.items() if isinstance(v, dict)}
    assert described == {"time", "plasma_radioactivity", "whole_blood_radioactivity"}

    # The flags follow from what is present, not from what they claimed.
    assert written["PlasmaAvail"] is True
    assert written["WholeBloodAvail"] is True
    assert written["MetaboliteAvail"] is False

    # Descriptions are the schema's, and units survive.
    assert "plasma" in written["plasma_radioactivity"]["Description"].lower()
    assert written["whole_blood_radioactivity"]["Units"] == "kBq/mL"

    # "Time" described a column named Time; the column is "time". A description
    # of something that is not there is the defect, so it goes.
    assert "Time" not in written

    # What they said that was not a column description is acquisition fact, and
    # they know things we did not ask them about, so it is kept.
    assert written["MetaboliteMethod"] == "HPLC"


# --------------------------------------------------------------------------
# real PMOD exports
# --------------------------------------------------------------------------


@real_blood
def test_whole_blood_and_plasma_convert(tmp_path: Path) -> None:
    src = BLOOD_DATA / "Ex_bld_wholeblood_and_plasma_only"
    basename = "sub-001_trc-FDG_run-1_pet"
    staging = _staged(tmp_path, basename)
    task = _task(
        basename,
        [
            (blood_role("wholeblood", "manual"), str(src / "whole_blood.bld")),
            (blood_role("plasma", "manual"), str(src / "plasma_parent.bld")),
        ],
    )
    assert convert_blood_files(staging, [task]) == 2

    pet = staging / "pet"
    table = pet / "sub-001_trc-FDG_run-1_recording-manual_blood.tsv"
    sidecar = pet / "sub-001_trc-FDG_run-1_recording-manual_blood.json"
    assert table.is_file() and sidecar.is_file()

    with table.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    assert "time" in header
    assert "whole_blood_radioactivity" in header
    assert "plasma_radioactivity" in header
    assert len(rows) > 1

    described = {
        k
        for k, v in json.loads(sidecar.read_text(encoding="utf-8")).items()
        if isinstance(v, dict)
    }
    assert described == set(header)


@real_blood
def test_mixed_sampling_splits_into_two_recordings(tmp_path: Path) -> None:
    """Hand-drawn and autosampled series have different time resolution.

    BIDS separates them with the ``recording`` entity, and pet2bids works out
    which is which. It writes one sidecar for both, with no recording entity on
    it, which no table can inherit by name. Each table therefore gets its own
    copy, and each copy describes the columns of both, since that is what the
    one sidecar they wrote described.
    """
    src = BLOOD_DATA / "Ex_bld_manual_and_autosampled_mixed"
    basename = "sub-002_ses-baseline_trc-PIB_pet"
    staging = _staged(tmp_path, basename)
    task = _task(
        basename,
        [
            (
                blood_role("wholeblood", "automatic"),
                str(src / "S001_Whole Blood Activity.bld"),
            ),
            (blood_role("plasma", "manual"), str(src / "S001_Plasma Activity.bld")),
            (
                blood_role("parentfraction", "manual"),
                str(src / "S001_ParentFraction.bld"),
            ),
        ],
    )
    # Two tables and a sidecar apiece.
    assert convert_blood_files(staging, [task]) == 4

    pet = staging / "pet"
    stem = "sub-002_ses-baseline_trc-PIB"
    assert (pet / f"{stem}_recording-manual_blood.tsv").is_file()
    assert (pet / f"{stem}_recording-automatic_blood.tsv").is_file()
    assert not (pet / f"{stem}_blood.json").exists()

    columns: set[str] = set()
    for table in pet.glob("*_blood.tsv"):
        with table.open(encoding="utf-8") as handle:
            columns.update(next(csv.reader(handle, delimiter="\t"), []))

    for recording in ("manual", "automatic"):
        sidecar = pet / f"{stem}_recording-{recording}_blood.json"
        written = json.loads(sidecar.read_text(encoding="utf-8"))
        assert {k for k, v in written.items() if isinstance(v, dict)} == columns
        assert written["MetaboliteAvail"] is True


# --------------------------------------------------------------------------
# what the template asks about blood
# --------------------------------------------------------------------------


def _inventory(companions=None):
    import pandas as pd

    return pd.DataFrame([{
        "include": "1",
        "proposed_datatype": "pet",
        "bids_guess_suffix": "pet",
        "companion_files": json.dumps(companions) if companions else "",
    }])


def test_the_availability_flags_are_never_asked() -> None:
    """They follow from what was attached, so a stated answer could only lie.

    We write them into the sidecar from the table we produced. A template
    answer saying otherwise would be overwritten at best and believed at worst.
    """
    from bidsmgr.metadata.template_plan import sidecar_section

    asked = {f.name for f in sidecar_section("pet", "blood").fields}
    assert not asked & {"WholeBloodAvail", "PlasmaAvail", "MetaboliteAvail"}

    # ... and the ones that ARE opinions still are.
    assert "DispersionCorrected" in asked
    assert "WithdrawalRate" in asked


def test_attaching_blood_makes_the_section_exist() -> None:
    """The gap this closes.

    A blood file is not a scanned row, so nothing in the inventory's datatype
    and suffix columns announced it, and the questions for it, including the
    REQUIRED DispersionCorrected, were asked nowhere at all.
    """
    from bidsmgr.metadata.template_plan import pair_counts, present_pairs

    assert ("pet", "blood") not in present_pairs(_inventory())

    with_blood = _inventory([
        {"suffix": blood_role("plasma", "manual"), "path": "/tmp/p.bld"},
    ])
    assert ("pet", "blood") in present_pairs(with_blood)
    assert pair_counts(with_blood)[("pet", "blood")] == 1


def test_an_excluded_row_brings_no_blood_section() -> None:
    from bidsmgr.metadata.template_plan import present_pairs

    df = _inventory([{"suffix": blood_role("plasma", "manual"), "path": "/tmp/p.bld"}])
    df.loc[0, "include"] = "0"
    assert ("pet", "blood") not in present_pairs(df)


def test_metabolite_questions_appear_only_when_metabolites_do() -> None:
    """A conditional requirement asked exactly when its condition holds.

    The schema marks MetaboliteMethod required only when metabolite data is
    available, so it is normally left out rather than demanded of everyone.
    Attaching a parent-fraction curve is that condition, and it is also what
    pet2bids warns about on the console after the fact.
    """
    from bidsmgr.metadata.template_plan import blood_conditions, sidecar_section

    plasma_only = _inventory([
        {"suffix": blood_role("plasma", "manual"), "path": "/tmp/p.bld"},
    ])
    assert blood_conditions(plasma_only) == ()

    with_metabolites = _inventory([
        {"suffix": blood_role("parentfraction", "manual"), "path": "/tmp/pf.bld"},
    ])
    conditions = blood_conditions(with_metabolites)
    assert "MetaboliteMethod" in conditions

    asked = {
        f.name for f in sidecar_section("pet", "blood", also_ask=conditions).fields
    }
    assert "MetaboliteMethod" in asked
    assert "MetaboliteRecoveryCorrectionApplied" in asked


def test_malformed_companion_json_is_not_a_crash() -> None:
    """A companion list is user data and reaches this from a spreadsheet."""
    from bidsmgr.metadata.template_plan import blood_conditions, present_pairs

    df = _inventory()
    df.loc[0, "companion_files"] = "{not json"
    assert ("pet", "blood") not in present_pairs(df)
    assert blood_conditions(df) == ()

"""The BIDS preview predicts what the conversion and the metadata step write,
with the user's answers in it.

The multimodal sample (MRI, spectroscopy, PET, EEG, MEG, eye tracking) is
scanned with a probe conversion, as the GUI scans by default, given a template
answer, a dataset-wide answer, the dataset's authors and licence and a
participants file, and converted through the chain the GUI runs after a
conversion (metadata with TODOs at the recommended scope, companion stubs, the
citation file). Then every predicted file must be written, every written file
predicted, and every value the preview states (anything but "…") must be the
value in the file, TODO placeholders included.

Gated on ``BIDS_MANAGER_REAL_MRI_DATA``; the sample lives beside the raw data
(``sample_datasets/2026-10/BIDS_Manager_multimodal_sample_v2``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.data_root import data_root

_ROOT = data_root()
SAMPLE = (_ROOT.parent / "sample_datasets" / "2026-10" / "BIDS_Manager_multimodal_sample_v2"
          if _ROOT is not None else None)

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not os.environ.get("BIDS_MANAGER_REAL_MRI_DATA"),
                       reason="BIDS_MANAGER_REAL_MRI_DATA not set"),
    pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(),
                       reason=f"multimodal sample missing at {SAMPLE}"),
]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    from bidsmgr.cli.convert import run_convert
    from bidsmgr.cli.metadata import run_metadata_cli
    from bidsmgr.cli.scan import run_scan
    from bidsmgr.recording_meta import dump_spec, load_spec, scaffold_sidecar_path

    work = tmp_path_factory.mktemp("preview")
    inv = work / "inv.tsv"
    run_scan(SAMPLE / "raw", inv, n_jobs=2, probe_convert=True)
    spec = load_spec(scaffold_sidecar_path(inv))
    spec.sequence_templates["func/bold"] = {"TaskDescription": "Rest with eyes open"}
    spec.defaults.institution_name = "University of Oldenburg"
    spec.dataset_description.authors = ["Ada Lovelace", "Alan Turing"]
    spec.dataset_description.license = "CC-BY-4.0"
    spec.participants_file = str(SAMPLE / "participants.csv")
    scaffold_sidecar_path(inv).write_text(dump_spec(spec), encoding="utf-8")
    assert run_convert(inv, work / "bids", raw_root=SAMPLE / "raw") == 0
    assert run_metadata_cli(work / "bids", inventory_tsv=inv, fill_scope="recommended",
                            generate_companions=True, write_citation_file=True) == 0
    root = next(p for p in (work / "bids").iterdir() if p.is_dir())
    return inv, root


def _plan(inv: Path, root: Path):
    from bidsmgr.classifier.acquisition import cache
    from bidsmgr.converter.output_plan import PlanOptions, plan_output
    from bidsmgr.recording_meta import load_spec, scaffold_sidecar_path

    df = pd.read_csv(inv, sep="\t", dtype=str, keep_default_na=False)
    spec = load_spec(scaffold_sidecar_path(inv))
    features = cache.read(inv)
    options = PlanOptions(dataset=root.name, inventory_dir=str(inv.parent),
                          fill_scope="recommended", companions=True, citation=True)
    return df, spec, features, options, plan_output(df, None, options, features, spec)


def test_the_files(converted) -> None:
    inv, root = converted
    *_rest, plan = _plan(inv, root)
    written = {p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and ".bidsmgr" not in p.parts}
    sure = {f.path for f in plan.files if not f.conditional}
    maybe = {f.path for f in plan.files if f.conditional}
    assert sure - written == set(), "predicted and not written"
    assert written - sure - maybe == set(), "written and not predicted"


def test_every_stated_value(converted) -> None:
    from bidsmgr.converter.output_plan import UNKNOWN, approximate

    inv, root = converted
    df, spec, features, options, plan = _plan(inv, root)
    wrong: list[str] = []
    for f in plan.files:
        if not f.path.endswith(".json") or not (root / f.path).is_file():
            continue
        got = json.loads((root / f.path).read_text())
        predicted = approximate(df, f, options, features, spec)
        for key, value, source in predicted.fields:
            if UNKNOWN in json.dumps(value, ensure_ascii=False):
                continue
            if got.get(key, "<absent>") != value:
                wrong.append(f"{f.path}: {key} predicted {value!r} ({source}), "
                             f"written {got.get(key, '<absent>')!r}")
        todo = {k for k, v in got.items() if v in ("TODO", ["TODO"])}
        marked = {k for k, v, s in predicted.fields if s == "todo" and v in ("TODO", ["TODO"])}
        if todo != marked:
            wrong.append(f"{f.path}: TODO written {sorted(todo)}, predicted {sorted(marked)}")
    assert not wrong, "\n".join(wrong[:20])


def test_the_users_answers_are_where_they_said(converted) -> None:
    from bidsmgr.converter.output_plan import approximate

    inv, root = converted
    df, spec, features, options, plan = _plan(inv, root)
    bold = next(f for f in plan.files if f.path.endswith("_task-rest_bold.json"))
    fields = {k: (v, s) for k, v, s in approximate(df, bold, options, features, spec).fields}
    assert fields["TaskDescription"] == ("Rest with eyes open", "stated")
    description = json.loads((root / "dataset_description.json").read_text())
    assert description["License"] == "CC-BY-4.0" and "Authors" not in description, \
        "the citation file holds the authors"
    assert "Lovelace" in (root / "CITATION.cff").read_text()


def _tables(converted):
    from bidsmgr.converter.output_plan import approximate
    from bidsmgr.metadata.tables import read_table

    inv, root = converted
    df, spec, features, options, plan = _plan(inv, root)
    maybe = {f.path for f in plan.files if f.conditional}
    out = []
    for f in plan.files:
        if not f.path.endswith(("_scans.tsv", "_sessions.tsv")):
            continue
        header, rows = read_table(root / f.path)
        written = {r[header[0]]: r.get("acq_time", "") for r in rows}
        said = dict(approximate(df, f, options, features, spec).rows)
        out.append((f.path, said, written, maybe))
    return out


def test_the_scans_and_sessions_tables_row_for_row(converted) -> None:
    """Every row the preview predicts is a row written: the preview asks the
    metadata step's own rule. A file the plan marks conditional (the second
    eye of an eye tracker) may also get its row."""
    tables = _tables(converted)
    assert any(path.endswith("_sessions.tsv") for path, *_ in tables), "sessions predicted"
    wrong = []
    for path, said, written, maybe in tables:
        home = path.rsplit("/", 1)[0]
        missing = set(said) - set(written)
        extra = {n for n in set(written) - set(said) if f"{home}/{n}" not in maybe}
        if missing or extra:
            wrong.append(f"{path}: predicted, not written {sorted(missing)}; "
                         f"written, not predicted {sorted(extra)}")
    assert not wrong, "\n".join(wrong)


def test_the_predicted_times_are_the_written_ones(converted) -> None:
    """Within the second or two the preview's note allows (the scan's time
    against the first volume's, which dcm2niix writes). The scan kept the
    time of whichever DICOM file it read first, and a multi-volume DWI came
    out a minute late; it keeps the earliest now (2026-10-19)."""
    from datetime import datetime

    from bidsmgr.converter.output_plan import UNKNOWN

    def seconds(stamp: str) -> float:
        return datetime.fromisoformat(stamp.rstrip("Z")).timestamp()

    wrong = []
    for path, said, written, _maybe in _tables(converted):
        for name, when in said.items():
            got = written.get(name, "")
            if when == UNKNOWN or not got or got == "n/a":
                continue
            if abs(seconds(when) - seconds(got)) > 3:
                wrong.append(f"{path}: {name} predicted {when}, written {got}")
    assert not wrong, "\n".join(wrong)

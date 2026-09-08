"""Turn a PET run's blood curves into ``_blood.tsv`` and ``_blood.json``.

Quantitative PET rests on the arterial input function: what the tracer was doing
in the blood while the scanner was counting it in tissue. BIDS has a place for
it, and until now BIDS Manager had nothing to put there.

The parsing is pet2bids'. Their ``PmodToBlood`` reads PMOD ``.bld`` exports,
scales time to seconds, works out which curves were drawn by hand and which came
off an autosampler, and writes the BIDS tables. It is the reference
implementation from the group that wrote this part of the standard, and
reimplementing it to avoid a dependency would be work spent going backwards.

What this module adds is everything around it:

* **It never asks.** ``PmodToBlood`` calls ``input()`` when it cannot tell manual
  sampling from automatic, which in a GUI is a window that hangs with no visible
  cause. Every series here carries its method, chosen by the user in the form, so
  the prompt is never reached.
* **The names are the run's own.** Left to itself it names files after the output
  directory, so a study with two tracers and three runs per subject would write
  the same filename repeatedly. Here the name comes from the same schema-driven
  basename the recording itself got, so ``trc-`` and ``run-`` are where BIDS
  expects them, and the ``recording-manual`` / ``recording-autosampler`` entity
  it works out is preserved.
* **It fits the row model.** Blood belongs to ONE PET run, which is what the
  inventory's companion files already express, so it needs no new concept.

Nothing here writes metadata the user did not supply. Whether the plasma was
dispersion-corrected, what the withdrawal rate was: those are sidecar fields and
they go through the template like every other field.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from ..util.pet2bids import quiet_pet2bids, telemetry_off

log = logging.getLogger(__name__)

# A companion entry tagged for blood, as ``blood:<series>:<method>``.
BLOOD_ROLE_PREFIX = "blood:"

# Our series names, and the argument pet2bids takes for each.
BLOOD_SERIES: dict[str, str] = {
    "wholeblood": "whole_blood_activity",
    "plasma": "plasma_activity",
    "parentfraction": "parent_fraction",
}

# How a sample was drawn. BIDS turns this into the ``recording`` entity, because
# a hand-drawn series and an autosampled one have different time resolution and
# belong in different files.
BLOOD_METHODS: frozenset[str] = frozenset({"manual", "automatic"})


def blood_role(series: str, method: str) -> str:
    """The companion tag for one blood curve."""
    return f"{BLOOD_ROLE_PREFIX}{series}:{method}"


def parse_blood_role(tag: str) -> Optional[tuple[str, str]]:
    """``(series, method)`` for a blood companion tag, else ``None``.

    Anything malformed returns ``None`` rather than raising: a companion list is
    user data, and one bad entry should not stop a conversion.
    """
    if not tag.startswith(BLOOD_ROLE_PREFIX):
        return None
    parts = tag[len(BLOOD_ROLE_PREFIX):].split(":")
    if len(parts) != 2:
        return None
    series, method = parts
    if series not in BLOOD_SERIES or method not in BLOOD_METHODS:
        return None
    return series, method


def is_blood_role(tag: str) -> bool:
    """True for a companion this module owns, so the copier leaves it alone."""
    return tag.startswith(BLOOD_ROLE_PREFIX)


def _blood_companions(task) -> dict[str, tuple[str, Path]]:
    """This run's blood curves, as ``{series: (method, path)}``."""
    found: dict[str, tuple[str, Path]] = {}
    for tag, src in getattr(task, "companion_files", ()) or ():
        parsed = parse_blood_role(str(tag))
        if parsed is None:
            continue
        series, method = parsed
        path = Path(src)
        if not path.is_file():
            log.warning("blood: source not found %s (%s); skipped", path, series)
            continue
        found[series] = (method, path)
    return found


def _entity_prefix(basename: str) -> str:
    """The run's name without its suffix, e.g. ``sub-001_trc-FDG_run-1``.

    Blood files carry the same entities as the recording they belong to, which
    is why this is taken from the basename the schema built rather than
    assembled here.
    """
    return basename.rsplit("_", 1)[0] if "_" in basename else basename


def _recording_entity(produced_name: str) -> str:
    """The ``recording-<label>`` part pet2bids chose, if it chose one.

    It decides this from how each series was sampled, and it is the one piece of
    the filename that is theirs to determine rather than ours.
    """
    for part in produced_name.split("_"):
        if part.startswith("recording-"):
            return part
    return ""


def _extension(name: str) -> str:
    return ".tsv" if name.endswith(".tsv") else ".json"


def convert_blood_files(subject_staging_dir: Path, tasks) -> int:
    """Convert each PET run's linked blood curves into the staged tree.

    Returns how many files were written. Best-effort throughout: a missing
    source, an unreadable curve or an absent dependency is logged and skipped,
    never raised, because a blood problem must not cost a user their imaging
    conversion.
    """
    if not subject_staging_dir.is_dir():
        return 0

    n_written = 0
    for task in tasks:
        curves = _blood_companions(task)
        if not curves:
            continue

        basename = str(getattr(task, "basename", "") or "")
        if not basename:
            continue
        sidecar = next(iter(subject_staging_dir.rglob(f"{basename}.json")), None)
        if sidecar is None:
            log.warning("blood: no staged sidecar for %s; skipped", basename)
            continue

        n_written += _convert_one(curves, sidecar, _entity_prefix(basename))
    return n_written


def _convert_one(curves: dict, sidecar: Path, prefix: str) -> int:
    """Convert one run's curves and file the results beside its sidecar."""
    converter = _load_converter()
    if converter is None:
        return 0

    kwargs: dict = {}
    for series, (method, path) in curves.items():
        argument = BLOOD_SERIES[series]
        kwargs[argument] = path
        # Always stated. Left unset, pet2bids asks on stdin and a GUI hangs.
        kwargs[f"{argument}_collection_method"] = method

    with tempfile.TemporaryDirectory() as scratch:
        # It names its output after the directory, so the directory is named
        # after the run. The real naming happens on the way out.
        staged = Path(scratch) / prefix
        staged.mkdir(parents=True, exist_ok=True)
        try:
            with quiet_pet2bids():
                converter(output_path=staged, output_json=True, **kwargs)
        except Exception as exc:  # noqa: BLE001 - never lose the imaging run
            log.warning("blood: conversion failed for %s: %s", prefix, exc)
            return 0

        # Their data dictionary does not always match the table beside it: in
        # 1.5.1, converting whole blood plus plasma describes a
        # metabolite_parent_fraction column that is not in the TSV and omits the
        # plasma_radioactivity column that is. A sidecar that describes columns
        # the file does not have, and misses ones it does, fails validation.
        # So the dictionary is rebuilt from the table that was actually written.
        _rewrite_dictionaries(staged)

        produced_files = sorted(staged.rglob("*_blood.*"))
        # BIDS makes ``recording`` a required entity on blood files, tables and
        # sidecars alike. pet2bids 1.5.1 puts it on the table and leaves it off
        # the sidecar, which pairs a valid table with an invalid sidecar. The
        # entity belongs to the sampling, so it is taken from the tables, and a
        # sidecar that arrived without one is written once per sampling it
        # describes.
        recordings = sorted({
            _recording_entity(p.name) for p in produced_files
            if p.name.endswith(".tsv") and _recording_entity(p.name)
        })

        written = 0
        for produced in produced_files:
            own = _recording_entity(produced.name)
            targets = [own] if own else (recordings or [""])
            for recording in targets:
                parts = [prefix] + ([recording] if recording else []) + ["blood"]
                dest = sidecar.with_name(
                    "_".join(parts) + _extension(produced.name))
                try:
                    shutil.copyfile(produced, dest)
                except OSError as exc:  # noqa: BLE001
                    log.warning("blood: could not write %s: %s", dest.name, exc)
                    continue
                written += 1
                log.info("blood: %s -> %s", produced.name, dest.name)
        return written


# The units BIDS names in its own column descriptions. The schema states them
# in prose rather than in a unit field, so they are written out here, and only
# used where pet2bids did not already say.
_COLUMN_UNITS: dict[str, str] = {
    "whole_blood_radioactivity": "kBq/mL",
    "plasma_radioactivity": "kBq/mL",
    "metabolite_parent_fraction": "arbitrary",
    "metabolite_polar_fraction": "arbitrary",
    "time": "s",
}

# Which column proves which availability flag. BIDS asks the sidecar to say what
# is present, and what is present is exactly what the table holds.
_AVAIL_FLAGS: dict[str, str] = {
    "whole_blood_radioactivity": "WholeBloodAvail",
    "plasma_radioactivity": "PlasmaAvail",
    "metabolite_parent_fraction": "MetaboliteAvail",
}


def _column_description(column: str) -> str:
    """What the standard says this column is, or nothing."""
    from ..schema.loader import get_schema

    try:
        defined = get_schema().objects.columns
    except Exception:  # noqa: BLE001
        return ""
    for entry in defined.values():
        if str(entry.get("name", "")) == column:
            return " ".join(str(entry.get("description", "")).split())
    return ""


def _rewrite_dictionaries(staged: Path) -> None:
    """Make each ``_blood.json`` describe the table it sits beside.

    Descriptions come from the schema, because what a column means is a fact
    about BIDS and facts about BIDS have one source. The availability flags
    follow from which columns exist. Anything pet2bids wrote that is not a
    column description is kept: it knows things about the acquisition we did not
    ask it about.
    """
    import csv

    for sidecar in staged.rglob("*_blood.json"):
        tables = sorted(sidecar.parent.glob("*_blood.tsv"))
        if not tables:
            continue
        columns: list[str] = []
        for table in tables:
            with table.open(encoding="utf-8") as handle:
                header = next(csv.reader(handle, delimiter="\t"), [])
            for name in header:
                if name and name not in columns:
                    columns.append(name)
        if not columns:
            continue

        try:
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}

        # Whatever it said that was not about a column: acquisition facts we
        # did not ask for and should not discard.
        rebuilt = {
            key: value
            for key, value in existing.items()
            if not isinstance(value, dict) and key not in _AVAIL_FLAGS.values()
        }
        for column in columns:
            said = existing.get(column)
            said = said if isinstance(said, dict) else {}
            entry = {
                "Description": _column_description(column)
                or said.get("Description", "")
                or f"{column} as recorded",
            }
            units = said.get("Units") or _COLUMN_UNITS.get(column)
            if units:
                entry["Units"] = units
            rebuilt[column] = entry
        for column, flag in _AVAIL_FLAGS.items():
            rebuilt[flag] = column in columns

        try:
            sidecar.write_text(
                json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:  # noqa: BLE001
            log.warning("blood: could not rewrite %s: %s", sidecar.name, exc)


def _load_converter():
    """``PmodToBlood``, or ``None`` with a reason logged.

    Telemetry is switched off before the import. The module we call carries
    none, but the package has some in its sibling modules, and a user who chose
    BIDS Manager cannot consent to reporting from a package they did not know
    they installed.
    """
    telemetry_off()
    try:
        from pypet2bids.convert_pmod_to_blood import PmodToBlood
    except Exception as exc:  # noqa: BLE001 - an odd install is not a crash
        log.warning(
            "blood: pypet2bids is unavailable (%s); blood curves were not "
            "converted. Everything else in this conversion is unaffected.", exc,
        )
        return None
    return PmodToBlood


__all__ = [
    "BLOOD_METHODS",
    "BLOOD_ROLE_PREFIX",
    "BLOOD_SERIES",
    "blood_role",
    "convert_blood_files",
    "is_blood_role",
    "parse_blood_role",
]

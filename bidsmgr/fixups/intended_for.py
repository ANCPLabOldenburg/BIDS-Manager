"""Link each fieldmap to the runs it corrects.

Two ways of saying it, both written by default:

* ``IntendedFor`` in the fieldmap's sidecar, listing its runs, as BIDS URIs
  (``bids::sub-X[/ses-Y]/<datatype>/<file>``) or, for tools that need the
  older form, paths relative to the subject (``[ses-Y/]<datatype>/<file>``,
  always with the session when there is one).
* ``B0FieldIdentifier`` on the fieldmap and ``B0FieldSource`` on each run it
  corrects.

Why both, measured on 2026-10-12 with sdcflows (what fMRIPrep uses to find
fieldmaps) and fMRIPrep's own lookup: a BIDS-URI ``IntendedFor`` links in
sdcflows 2.13 and 2.15 (fMRIPrep 25.1, 25.2) and is SILENTLY ignored by 2.5 to
2.10 (fMRIPrep 23.1 to 24.1), which then corrects nothing and says nothing.
The subject-relative form links in every version, the form without the
session in none, and ``B0FieldIdentifier`` / ``B0FieldSource`` in every
version, taking precedence. Record: workspace
``reference/FIELDMAP_LINKS_AND_FMRIPREP.md``.

Which runs: :func:`bind`, ONE rule for the conversion (on the staged
sidecars), the Editor's References tool and the Converter's preview (on the
inventory, :func:`plan_links`), so none can disagree with the others.

1. A fieldmap made for one kind of run (an EPI reference recorded with the
   diffusion sequence) only serves that kind; a gradient-echo fieldmap
   serves BOLD, diffusion and ASL runs alike.
2. The same ``ShimSetting`` is the strongest evidence: same shim, same B0.
3. Then time: the last fieldmap acquired before the run, else the first one
   after it.
4. Without times every run is offered every fieldmap, and the reason says
   so.

The user's own choice, made in the Converter (``fieldmap_targets``), wins.

One fieldmap is every file of one acquisition: a gradient echo's magnitude
and phase, the ``part-mag`` and ``part-phase`` of an EPI reference, and both
directions of a reversed-phase pair (``_dir-AP_epi`` and ``_dir-PA_epi``),
which are estimated together. A phase image never joins an EPI estimator.
A reversed-phase reference of ONE direction cannot be estimated alone:
sdcflows needs two directions under one ``B0FieldIdentifier``, drops a group
that has one, and once any group works it no longer falls back to
``IntendedFor``, so the run silently went uncorrected. The nearest run
phase-encoded the other way (its SBRef when it has one) joins the group.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from ..inventory._time import parse_time_seconds

log = logging.getLogger(__name__)

#: How ``IntendedFor`` names a run.
FORMS = ("uri", "relative")

#: The runs a fieldmap can correct: datatype -> suffixes.
TARGETS: dict[str, tuple[str, ...]] = {"func": ("bold",), "dwi": ("dwi",), "perf": ("asl",)}

_FMAP_SUFFIX_STRIP_RE = re.compile(
    r"_(magnitude[12]?|phasediff|phase[12]|fieldmap|epi)$"
)
_PART_RE = re.compile(r"_part-([^_]+)")
_DIR_RE = re.compile(r"_dir-[^_]+")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fieldmap:
    """One fieldmap acquisition (its magnitude and phase files share it)."""

    key: str
    time: Optional[float] = None
    shim: Optional[tuple[float, ...]] = None
    #: The datatype it was recorded for, when it can only serve one
    #: (``dwi`` for a diffusion EPI reference); ``None``: any.
    serves: Optional[str] = None
    members: tuple = field(default=(), compare=False)


@dataclass(frozen=True)
class Run:
    key: str
    datatype: str
    time: Optional[float] = None
    shim: Optional[tuple[float, ...]] = None


def _same_shim(a: Optional[tuple], b: Optional[tuple]) -> bool:
    return a is not None and b is not None and len(a) == len(b) and all(
        abs(x - y) <= 1e-3 * max(1.0, abs(x)) for x, y in zip(a, b))


def bind(fieldmaps: Iterable[Fieldmap], runs: Iterable[Run]) -> tuple[dict[str, list[str]], str]:
    """``{fieldmap key: [run keys]}`` and one sentence saying why."""
    fieldmaps, runs = list(fieldmaps), list(runs)
    out: dict[str, list[str]] = {f.key: [] for f in fieldmaps}
    if not fieldmaps or not runs:
        return out, ""
    by_shim = by_time = fallback = 0
    for run in runs:
        eligible = [f for f in fieldmaps if f.serves in (None, run.datatype)]
        if not eligible:
            continue
        same = [f for f in eligible if _same_shim(f.shim, run.shim)]
        if same:
            eligible = same
        timed = run.time is not None and all(f.time is not None for f in eligible)
        if len(eligible) == 1:
            chosen = eligible
            by_shim += bool(same)
            by_time += not same
        elif timed:
            before = [f for f in eligible if f.time <= run.time]
            after = [f for f in eligible if f.time > run.time]
            chosen = [max(before, key=lambda f: f.time)] if before else \
                [min(after, key=lambda f: f.time)]
            by_shim += bool(same)
            by_time += not same
        else:
            chosen = eligible
            fallback += 1
        for f in chosen:
            out[f.key].append(run.key)
    parts = []
    if by_shim:
        parts.append(f"{by_shim} run(s) by their shim settings")
    if by_time:
        parts.append(f"{by_time} run(s) by acquisition times (each to the last "
                     "fieldmap before it)")
    if fallback:
        parts.append(f"{fallback} run(s) {FALLBACK_NOTE}")
    return out, "linked " + "; ".join(parts) if parts else ""


#: How the reason says a run was linked to every fieldmap, for want of times:
#: a guess, which the Editor's coherence check does not hold a dataset to.
FALLBACK_NOTE = "offered to every fieldmap, for lack of acquisition times"


def serves_of(meta: Mapping, has_gradients: bool = False) -> Optional[str]:
    """The datatype an EPI fieldmap was recorded for, from its sidecar:
    diffusion when it carries gradients or its sequence is a diffusion one,
    functional for a BOLD sequence, ASL for a labelling one; ``None`` for a
    gradient-echo fieldmap, which serves every kind."""
    if has_gradients:
        return "dwi"
    text = " ".join(str(meta.get(k, "")) for k in (
        "PulseSequenceDetails", "SequenceName", "SeriesDescription", "ProtocolName")).lower()
    if not text.strip() or "fm2d" in text or "gre_field" in text:
        return None
    if any(t in text for t in ("ep_b", "diff", "dwi", "dti")):
        return "dwi"
    if any(t in text for t in ("asl", "pcasl", "pasl")):
        return "perf"
    if any(t in text for t in ("epfid", "bold", "fmri", "mbep2d")):
        return "func"
    return None


def shim_of(meta: Mapping) -> Optional[tuple[float, ...]]:
    raw = meta.get("ShimSetting")
    if isinstance(raw, (list, tuple)) and raw:
        try:
            return tuple(float(x) for x in raw)
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# On the staged (or converted) files
# ---------------------------------------------------------------------------


def populate_intended_for(
    subject_staging_dir: Path,
    subject: str,
    session: Optional[str] = None,
    *,
    form: str = "uri",
    b0_field: bool = True,
    links: Optional[Mapping[str, list[str]]] = None,
) -> int:
    """Link every fieldmap under ``subject_staging_dir`` to its runs.

    ``form`` is how ``IntendedFor`` names a run (:data:`FORMS`); ``b0_field``
    also writes ``B0FieldIdentifier`` / ``B0FieldSource``; ``links`` is the
    user's own choice, ``{fieldmap image name: [run image names]}``, used
    instead of the rule for the fieldmaps it names. Returns the number of
    fieldmap sidecars written.
    """
    del session  # discovered from the layout
    if form not in FORMS:
        raise ValueError(f"IntendedFor form {form!r}: one of {FORMS}")
    if not subject_staging_dir.is_dir():
        return 0
    session_dirs = sorted(d for d in subject_staging_dir.glob("ses-*") if d.is_dir())
    roots = [(d, d.name[len("ses-"):]) for d in session_dirs] or [(subject_staging_dir, None)]
    n = 0
    for root, ses in roots:
        n += _link_one_root(root, subject, ses, form=form, b0_field=b0_field, links=links or {})
    return n


def suggest_intended_for(
    root: Path, subject: str, session: Optional[str],
) -> tuple[list[tuple[list[Path], list[str]]], Optional[str]]:
    """What ``IntendedFor`` SHOULD say under ``root`` (BIDS URIs), without
    writing it: one ``(sidecars, uris)`` pair per fieldmap acquisition, and
    the reason. The Editor's References and coherence tools use it, so they
    propose exactly what the conversion writes."""
    groups, runs, why = _plan_root(root, links={})
    if not groups:
        return [], None
    pairs = [(list(members), [_name_run(subject, session, run, "uri") for run in targets])
             for members, targets in groups]
    return pairs, why or None


def _link_one_root(root: Path, subject: str, session: Optional[str], *, form: str,
                   b0_field: bool, links: Mapping[str, list[str]]) -> int:
    groups, runs, _why = _plan_root(root, links)
    if not groups:
        return 0
    n = 0
    sources: dict[Path, list[str]] = {}
    for idx, (members, targets) in enumerate(groups, start=1):
        names = [_name_run(subject, session, run, form) for run in targets]
        ident = fieldmap_identifier(session, idx)
        for sidecar in members:
            meta = _load_json(sidecar)
            if names and not _is_phase_part(sidecar.name):
                meta["IntendedFor"] = names
                if b0_field:
                    meta["B0FieldIdentifier"] = ident
            else:
                # Nothing to correct: no key at all. An empty list claims
                # "intended for nothing", and an identifier no run names
                # makes fMRIPrep estimate a fieldmap it never uses.
                meta.pop("IntendedFor", None)
                meta.pop("B0FieldIdentifier", None)
            if _write_json(sidecar, meta):
                n += 1
        if b0_field and names:
            for run in targets:
                sources.setdefault(_matching_json(run), []).append(ident)
            partner = _second_direction(members, targets)
            if partner is not None:
                _add_identifier(partner, ident)
                log.info("fieldmap %s has one phase-encoding direction: %s joins %s",
                         members[0].name, partner.name, ident)
        log.info("fieldmap %s: %d run(s)%s", members[0].name if members else "?", len(names),
                 f", {ident}" if b0_field else "")
    for sidecar, idents in sources.items():
        meta = _load_json(sidecar)
        meta["B0FieldSource"] = idents[0] if len(idents) == 1 else idents
        _write_json(sidecar, meta)
    return n


def _plan_root(root: Path, links: Mapping[str, list[str]]):
    """``([(fieldmap sidecars, [run images])...], runs, why)`` under one
    subject or session folder."""
    fmap_dir = root / "fmap"
    if not fmap_dir.is_dir():
        return [], [], ""
    runs = _collect_runs(root)
    if not runs:
        return [], [], ""
    grouped: dict[str, dict] = {}
    for sidecar in sorted(fmap_dir.glob("*.json")):
        if not _FMAP_SUFFIX_STRIP_RE.search(sidecar.name[: -len(".json")]):
            continue            # not a fieldmap file (a name BIDS does not define)
        key = _fieldmap_group_key(sidecar)
        meta = _load_json(sidecar)
        slot = grouped.setdefault(key, {"members": [], "time": None, "shim": None,
                                        "serves": None, "images": []})
        slot["members"].append(sidecar)
        image = _image_for(sidecar)
        if image is not None:
            slot["images"].append(image.name)
        when = _time_of(meta)
        if when is not None and (slot["time"] is None or when < slot["time"]):
            slot["time"] = when
        slot["shim"] = slot["shim"] or shim_of(meta)
        gradients = sidecar.with_suffix(".bval").is_file()
        slot["serves"] = slot["serves"] or serves_of(meta, gradients)
    if not grouped:
        return [], [], ""
    fieldmaps = [Fieldmap(key=k, time=s["time"], shim=s["shim"], serves=s["serves"])
                 for k, s in grouped.items()]
    by_key = {str(img): (img, dt) for img, dt in runs}
    run_objs = []
    for image, datatype in runs:
        meta = _load_json(_matching_json(image))
        run_objs.append(Run(key=str(image), datatype=datatype, time=_time_of(meta),
                            shim=shim_of(meta)))
    binding, why = bind(fieldmaps, run_objs)
    # The user's own choice wins, fieldmap by fieldmap.
    name_to_image = {img.name: img for img, _dt in runs}
    for key, slot in grouped.items():
        chosen = next((links[n] for n in slot["images"] if n in links), None)
        if chosen is not None:
            binding[key] = [str(name_to_image[n]) for n in chosen if n in name_to_image]
            why = (why + "; " if why else "") + "your own choice for some fieldmaps"
    groups = []
    for key, slot in sorted(grouped.items()):
        targets = [by_key[k][0] for k in binding.get(key, []) if k in by_key]
        targets.sort(key=lambda p: (p.parent.name, p.name))
        groups.append((slot["members"], targets))
    return groups, runs, why


def _collect_runs(root: Path) -> list[tuple[Path, str]]:
    out = []
    for datatype, suffixes in TARGETS.items():
        folder = root / datatype
        if not folder.is_dir():
            continue
        for image in sorted(folder.iterdir()):
            name = image.name
            if not image.is_file() or not name.endswith((".nii", ".nii.gz")):
                continue
            stem = re.sub(r"\.nii(\.gz)?$", "", name.lower())
            if "_bold_aborted" in stem or "_sbref" in stem:
                continue
            if stem.rsplit("_", 1)[-1] in suffixes:
                out.append((image, datatype))
    return out


def _name_run(subject: str, session: Optional[str], image: Path, form: str) -> str:
    parts = ([f"ses-{session}"] if session else []) + [image.parent.name, image.name]
    if form == "relative":
        return "/".join(parts)
    return "bids::" + "/".join([f"sub-{subject}", *parts])


def _time_of(meta: Mapping) -> Optional[float]:
    return parse_time_seconds(meta.get("AcquisitionTime") or meta.get("acq_time")
                              or meta.get("AcquisitionDateTime"))


#: The kind of fieldmap each suffix belongs to; sdcflows never mixes kinds.
_KIND = {"epi": "epi", "fieldmap": "fieldmap", "magnitude": "fieldmap"}


def fieldmap_identifier(session: Optional[str], index: int) -> str:
    """The ``B0FieldIdentifier`` of a session's ``index``-th fieldmap
    acquisition (from 1), counting them in :func:`group_key` order, the ones
    that correct no run included."""
    return f"fmap{re.sub(r'[^0-9A-Za-z]', '', session or '')}{index}"


def group_key(stem: str) -> str:
    """The fieldmap acquisition a file (or inventory name) belongs to: its
    name without the fieldmap suffix and ``part-``, and, for an EPI
    reference, without ``dir-`` (the two directions of a pair are one
    fieldmap), qualified by its kind, so a ``_fieldmap`` and a
    ``_phasediff`` named alike stay two."""
    m = _FMAP_SUFFIX_STRIP_RE.search(stem)
    suffix = m.group(1) if m else ""
    kind = _KIND.get(suffix, "phase" if suffix else "")
    if kind == "epi":
        stem = _DIR_RE.sub("", stem)
    return f"{_FMAP_SUFFIX_STRIP_RE.sub('', _PART_RE.sub('', stem))}#{kind}"


def acquisitions(rows: Iterable[tuple]) -> list[list]:
    """``rows`` (``(key, name, time)``, one session's fieldmap rows) grouped
    into acquisitions by :func:`group_key`, in time order. Two rows with the
    SAME name are two acquisitions (an inventory can hold a name twice until
    the user resolves it; on disk it cannot)."""
    groups: list[tuple[str, list, set]] = []
    for key, name, _when in sorted(rows, key=lambda r: (r[2] is None, r[2] or 0.0)):
        gk = group_key(name) if name else f"#{key}"
        slot = next((g for g in groups if g[0] == gk and name not in g[2]), None)
        if slot is None:
            slot = (gk, [], set())
            groups.append(slot)
        slot[1].append(key)
        slot[2].add(name)
    return [members for _gk, members, _names in groups]


def _pe_axis(meta: Mapping) -> Optional[tuple[str, bool]]:
    """``("j", True)`` for ``j-``: the axis and whether it is reversed."""
    pe = str(meta.get("PhaseEncodingDirection") or "").strip()
    if not pe or pe[0] not in "ijk":
        return None
    return pe[0], pe.endswith("-")


def _second_direction(members: list[Path], targets: list[Path]) -> Optional[Path]:
    """For an EPI reference of one phase-encoding direction: the sidecar of
    the run that gives it the other one (the nearest in time of its runs
    phase-encoded the opposite way, its SBRef when it has one). ``None``
    when the fieldmap needs none or no run qualifies."""
    sources = [m for m in members if not _is_phase_part(m.name)]
    if not sources or not all(m.name.endswith("_epi.json") for m in sources):
        return None
    metas = [_load_json(m) for m in sources]
    axes = {_pe_axis(m) for m in metas}
    if len(axes) != 1 or None in axes:
        return None
    axis, reversed_ = axes.pop()
    when = min((t for t in (_time_of(m) for m in metas) if t is not None), default=None)
    best: Optional[tuple[float, Path]] = None
    for run in targets:
        if _is_phase_part(run.name):
            continue
        sidecar = _matching_json(run)
        meta = _load_json(sidecar)
        if _pe_axis(meta) != (axis, not reversed_):
            continue
        t = _time_of(meta)
        gap = abs(t - when) if t is not None and when is not None else float("inf")
        if best is None or gap < best[0]:
            best = (gap, _sbref_of(run) or sidecar)
    return best[1] if best else None


def _sbref_of(run: Path) -> Optional[Path]:
    """The SBRef sidecar recorded with ``run`` (same name, ``part-`` aside)."""
    stem = _PART_RE.sub("", re.sub(r"\.nii(\.gz)?$", "", run.name))
    stem = re.sub(r"_(bold|dwi|asl)$", "_sbref", stem)
    candidate = run.with_name(stem + ".json")
    return candidate if candidate.is_file() and \
        (_image_for(candidate) is not None) else None


def _add_identifier(sidecar: Path, ident: str) -> None:
    meta = _load_json(sidecar)
    have = meta.get("B0FieldIdentifier")
    if have is None or have == ident:
        meta["B0FieldIdentifier"] = ident
    elif isinstance(have, list):
        if ident not in have:
            meta["B0FieldIdentifier"] = [*have, ident]
    else:
        meta["B0FieldIdentifier"] = [have, ident]
    _write_json(sidecar, meta)


def _fieldmap_group_key(json_path: Path) -> str:
    stem = json_path.name[: -len(".json")] if json_path.name.endswith(".json") else json_path.name
    return group_key(stem)


def _is_phase_part(name: str) -> bool:
    """A phase (or real/imaginary) image of an EPI: never an estimator's
    source, so it carries no link sdcflows could pick up."""
    m = _PART_RE.search(name)
    return bool(m) and m.group(1) in ("phase", "real", "imag") and \
        re.sub(r"\.(json|nii(\.gz)?)$", "", name).endswith(("_epi", "_bold", "_dwi", "_sbref"))


def _image_for(sidecar: Path) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        candidate = sidecar.with_name(sidecar.name[: -len(".json")] + ext)
        if candidate.is_file():
            return candidate
    return None


def _matching_json(image: Path) -> Path:
    if image.name.endswith(".nii.gz"):
        return image.with_suffix("").with_suffix(".json")
    return image.with_suffix(".json")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}


def _write_json(path: Path, meta: dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=4)
            fh.write("\n")
        return True
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# On the inventory: the Converter's preview, before anything is converted
# ---------------------------------------------------------------------------


#: The inventory column holding the user's own choice for a fieldmap row:
#: a JSON list of the row ids it corrects ("" = the rule decides).
LINKS_COLUMN = "fieldmap_targets"


def is_target_row(datatype: str, suffix: str) -> bool:
    return suffix in TARGETS.get(datatype, ())


def plan_links(df, shims: Optional[Mapping[str, tuple]] = None) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Which runs each fieldmap row of an inventory corrects, by the same
    rule as the conversion: ``({fieldmap row: [run rows]}, {fieldmap row:
    why})``, within each subject and session. ``shims`` (series uid ->
    shim values) comes from the classifier's features when known."""
    shims = shims or {}

    def cell(i, col: str) -> str:
        if col not in df.columns:
            return ""
        v = df.at[i, col]
        return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()

    def suffix_of(i) -> str:
        return cell(i, "bids_guess_suffix") or cell(i, "bids_name").rsplit("_", 1)[-1]

    def included(i) -> bool:
        return cell(i, "include") not in ("0", "false", "False")

    sessions: dict[tuple[str, str], dict[str, list[int]]] = {}
    for i in df.index:
        if not included(i):
            continue
        key = (cell(i, "participant_id"), cell(i, "session"))
        slot = sessions.setdefault(key, {"fmaps": [], "runs": []})
        dt = cell(i, "datatype")
        if dt == "fmap":
            slot["fmaps"].append(int(i))
        elif is_target_row(dt, suffix_of(i)):
            slot["runs"].append(int(i))

    plan: dict[int, list[int]] = {}
    reasons: dict[int, str] = {}
    from ..project.orchestration import row_id

    def shim_of_row(i) -> Optional[tuple]:
        for uid in cell(i, "series_uid").split("|"):
            if shims.get(uid) is not None:
                return tuple(shims[uid])
        return None

    for slot in sessions.values():
        # One fieldmap per acquisition, as the conversion groups its files:
        # the magnitude and phase rows, the parts, both directions of a pair.
        fieldmaps = []
        for n, rows in enumerate(acquisitions(
                (i, cell(i, "bids_name"), parse_time_seconds(cell(i, "acq_time")))
                for i in slot["fmaps"])):
            key = f"acq{n}"
            times = [t for t in (parse_time_seconds(cell(i, "acq_time")) for i in rows)
                     if t is not None]
            serves = next((s for s in (serves_of({"SeriesDescription": cell(i, "sequence")})
                                       for i in rows if suffix_of(i) == "epi") if s), None)
            fieldmaps.append(Fieldmap(
                key=key, time=min(times) if times else None,
                shim=next((s for s in map(shim_of_row, rows) if s is not None), None),
                serves=serves, members=tuple(rows)))
        runs = [Run(key=str(i), datatype=cell(i, "datatype"),
                    time=parse_time_seconds(cell(i, "acq_time")),
                    shim=shim_of_row(i)) for i in slot["runs"]]
        binding, why = bind(fieldmaps, runs)
        ids = {row_id(df, i): i for i in slot["runs"]}
        for f in fieldmaps:
            # The user's choice on any row of the acquisition is the
            # acquisition's, as the conversion reads it.
            chosen = None
            for i in f.members:
                own = cell(i, LINKS_COLUMN)
                if not own:
                    continue
                try:
                    chosen = [ids[r] for r in json.loads(own) if r in ids]
                except (ValueError, TypeError):
                    chosen = None
                if chosen is not None:
                    break
            for i in f.members:
                if chosen is not None:
                    plan[i], reasons[i] = list(chosen), "your own choice"
                else:
                    plan[i] = [int(k) for k in binding.get(f.key, [])]
                    reasons[i] = _reason_for(f, plan[i], why, fieldmaps)
    return plan, reasons


def _reason_for(fieldmap: Fieldmap, runs: list, why: str, everyone: list) -> str:
    """One sentence about one fieldmap's runs, for the Converter."""
    if not runs:
        if fieldmap.serves:
            return (f"Nothing to correct: it was recorded for {fieldmap.serves} runs, and "
                    "none is closer to it than to another fieldmap.")
        return ("Nothing to correct: every run is closer in time (or in shim) to "
                "another fieldmap.")
    if "lack of acquisition times" in why:
        return "Offered every run of the session: the acquisition times are missing."
    if "shim" in why and fieldmap.shim is not None:
        return "The runs recorded with the same shim settings as this fieldmap."
    if len(everyone) == 1:
        return "The only fieldmap of the session: it corrects every run."
    return "The runs acquired after this fieldmap and before the next one."


__all__ = ["FORMS", "Fieldmap", "LINKS_COLUMN", "Run", "TARGETS", "acquisitions", "bind",
           "fieldmap_identifier",
           "group_key", "is_target_row", "plan_links", "populate_intended_for", "serves_of", "shim_of",
           "suggest_intended_for"]

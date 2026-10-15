"""The parameter-based classifier, inside the scan.

``cli/scan`` hands over its rows (one per DICOM series), the features its
own header pass read, what the existing chain chose and what dcm2niix's
guess was on its own. This scores every series, looks at each session as a
whole, decides, and (in ``applied`` mode) replaces the chain's choice where
the decision says so. In ``shadow`` mode nothing changes but the report.

The same function re-scores an inventory from its features cache
(``bidsmgr-classify``, the GUI's Re-classify), so the scan and a later
re-run cannot disagree. Qt-free.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..types import Classification
from .config import DEFAULT, ClassifierConfig
from .engine import Decision, Series, Verdict, classify_session, decide, domain_datatypes, \
    names_only, report, rulebook, same
from .features import SeriesFeatures

log = logging.getLogger(__name__)

#: The note a row gets when its name should be looked at.
CHECK_TOKEN = "check the name"


@dataclass
class Outcome:
    """One series' verdict and decision."""

    verdict: Verdict
    decision: Decision
    features: Optional[SeriesFeatures] = None

    def report(self) -> dict:
        return report(self.verdict, self.decision)


def _order(acq_time: Optional[str], index: int) -> float:
    try:
        return float(str(acq_time or "").replace(":", ""))
    except ValueError:
        return float(index)


def classify(series: list[dict], features: dict[str, SeriesFeatures], *,
             others: Optional[dict[str, str]] = None,
             cfg: ClassifierConfig = DEFAULT, book: Optional[dict] = None
             ) -> dict[str, Outcome]:
    """Score and decide ``series`` (each ``{"key", "uid", "session",
    "acq_time"}``; the session is any grouping key, a participant and
    session). ``others`` is dcm2niix's own guess per key
    (``"datatype/suffix"``). Returns ``{key: Outcome}`` for every series
    with features."""
    others = others or {}
    book = book or rulebook(cfg.rulebook)
    sessions: dict[str, list[Series]] = defaultdict(list)
    for i, s in enumerate(series):
        f = features.get(s.get("uid", ""))
        if f is None:
            continue
        sessions[str(s.get("session", ""))].append(
            Series(key=s["key"], features=f, order=_order(s.get("acq_time"), i)))
    out: dict[str, Outcome] = {}
    for members in sessions.values():
        classify_session(members, cfg, book)
        for x in members:
            if not x.verdict.probabilities:
                continue
            decision = decide(x.verdict, others.get(x.key, ""),
                              names_only(x.features, cfg, book), cfg)
            out[x.key] = Outcome(x.verdict, decision, x.features)
    return out


def _name_of(c: Optional[Classification]) -> str:
    if c is None:
        return ""
    if c.skip or c.datatype == "discard":
        return "exclude/" + (c.suffix or "discard")
    return f"{c.datatype}/{c.suffix}"


def _allowed(entities: dict, datatype: str, suffix: str) -> dict:
    from ... import schema

    try:
        allowed = set(schema.allowed_entities(datatype, suffix))
    except Exception:  # noqa: BLE001
        return dict(entities)
    return {k: v for k, v in entities.items() if k in allowed}


def apply(chosen: dict[str, Classification], outcomes: dict[str, Outcome], row_ids: dict,
          *, forced=frozenset()) -> dict[str, Classification]:
    """``chosen`` with the decisions laid over it (``applied`` mode). ``row_ids``
    maps an outcome key to its row's UUID; ``forced`` are the keys a user's
    forced rule named, which nothing overrides."""
    from .. import sequence_dict

    out = dict(chosen)
    domain = domain_datatypes()
    for key, outcome in outcomes.items():
        d = outcome.decision
        if key in forced or key not in row_ids or not d.name \
                or d.decided_by not in ("parameters", "agreement"):
            continue
        current = out.get(key)
        if current is not None and not (current.skip or current.datatype == "discard") \
                and (current.datatype not in domain or current.suffix == "physio"):
            continue            # mrs, pet, physio: another rule's, never ours
        datatype, _, suffix = d.name.partition("/")
        if datatype == "exclude":
            if _name_of(current).startswith("exclude/"):
                continue
            first = outcome.verdict.evidence_for(d.name, 1)
            out[key] = Classification(
                row_id=row_ids[key], classifier="acquisition", datatype="discard",
                suffix=suffix, confidence=d.confidence, skip=True,
                rationale="left out: " + (first[0].text if first else suffix))
            continue
        if current is not None and current.datatype == "fmap" and datatype == "fmap" \
                and same(d.name, _name_of(current)):
            continue            # dcm2niix splits the fieldmap; its names stand
        entities = dict(current.candidate_entities) if current is not None else {}
        if current is None or current.datatype != datatype:
            # A change of kind keeps the name's task and acquisition label,
            # nothing else that described another kind of file.
            entities = {k: v for k, v in entities.items() if k in ("task", "acquisition")}
        entities.update(d.entities)
        if d.name == "fmap/epi" and not entities.get("direction") \
                and outcome.features is not None and outcome.features.pe_dir:
            entities["direction"] = outcome.features.pe_dir
        if datatype == "func" and not entities.get("task"):
            # From the name, as the chain does; with none, the scan's own
            # placeholder asks the user for it.
            description = outcome.features.description if outcome.features else ""
            task = sequence_dict.guess_task_from_text(description)
            if task:
                entities["task"] = task
        entities = _allowed(entities, datatype, suffix)
        if current is not None and same(d.name, _name_of(current)):
            if entities == current.candidate_entities:
                continue
            out[key] = current.model_copy(update={"candidate_entities": entities})
            continue
        out[key] = Classification(
            row_id=row_ids[key], classifier="acquisition", datatype=datatype, suffix=suffix,
            candidate_entities=entities, confidence=d.confidence,
            rationale=d.note or "acquisition parameters")
    # A contrast agent is a fact of the acquisition, whoever named the
    # suffix: dcm2niix's T1w after an injection is ce- too.
    for key, outcome in outcomes.items():
        label = outcome.verdict.entities.get("ceagent")
        current = out.get(key)
        if not label or key in forced or current is None or current.skip \
                or current.datatype not in domain \
                or current.candidate_entities.get("ceagent") == label:
            continue
        entities = _allowed({**current.candidate_entities, "ceagent": label},
                            current.datatype, current.suffix)
        if "ceagent" in entities:
            out[key] = current.model_copy(update={"candidate_entities": entities})
    return out


def columns(outcome: Outcome) -> dict[str, str]:
    """The three inventory columns for one series."""
    import json

    rep = outcome.report()
    return {
        "classification_confidence": f"{rep['confidence']:.3f}",
        "classification_alternatives": json.dumps(rep["alternatives"]),
        "classification_evidence": json.dumps(rep["evidence"]),
    }


__all__ = ["CHECK_TOKEN", "Outcome", "apply", "classify", "columns"]


# ---------------------------------------------------------------------------
# Scoring an inventory again (bidsmgr-classify, the GUI's Re-classify)
# ---------------------------------------------------------------------------


@dataclass
class Change:
    """What re-scoring an inventory row would change."""

    index: int
    old: str                       # "datatype/suffix" now ("exclude/..." when left out)
    new: str                       # what the decision names
    entities: dict
    exclude: bool
    columns: dict                  # the three classification columns
    check: bool
    #: Whether the row's name changes (a new kind, or new entities such as
    #: ``part``, ``direction`` or ``inversion`` on the same kind).
    renamed: bool = False


def _current(row) -> str:
    datatype = str(row.get("datatype", "") or "")
    suffix = str(row.get("bids_guess_suffix", "") or "")
    if not datatype:
        return "exclude/" + (suffix or "?") if str(row.get("include", "1")) == "0" else ""
    return f"{datatype}/{suffix}"


def rescore(df, features: dict[str, SeriesFeatures], cfg: ClassifierConfig = DEFAULT,
            book: Optional[dict] = None) -> list[Change]:
    """Score the MRI rows of an inventory again from their cached features.
    dcm2niix's guess is the one the scan recorded in each row's evidence.
    Returns one :class:`Change` per scored row (``new == old`` when the name
    stays)."""
    import json

    series, others, rows = [], {}, {}
    for i, row in df.iterrows():
        uids = [u for u in str(row.get("series_uid", "") or "").split("|") if u]
        if not uids or str(row.get("modality", "mri")) not in ("mri", ""):
            continue
        key = str(i)
        rows[key] = row
        series.append({"key": key, "uid": uids[0],
                       "session": f"{row.get('participant_id', '')}|{row.get('session', '')}",
                       "acq_time": row.get("acq_time", "")})
        try:
            other = json.loads(str(row.get("classification_evidence", "") or "{}")).get("other")
        except ValueError:
            other = ""
        if other:
            others[key] = other
    outcomes = classify(series, features, others=others, cfg=cfg, book=book)
    domain = domain_datatypes(book or rulebook(cfg.rulebook))
    changes = []
    for key, outcome in outcomes.items():
        row = rows[key]
        d = outcome.decision
        old = _current(row)
        if old and not old.startswith("exclude/") and \
                (old.split("/", 1)[0] not in domain or old.endswith("/physio")):
            continue            # mrs, pet, physio: another rule's, never ours
        new = old
        entities = {}
        try:
            entities = json.loads(str(row.get("entities", "") or "{}"))
        except ValueError:
            entities = {}
        before = dict(entities)
        if d.name and d.decided_by in ("parameters", "agreement") and \
                not (old.startswith("fmap/") and same(d.name, old)):
            new = old if same(d.name, old) else d.name
            if not d.excluded:
                datatype, _, suffix = d.name.partition("/")
                if not old.startswith(datatype + "/"):
                    entities = {k: v for k, v in entities.items()
                                if k in ("subject", "session", "task", "acquisition", "run")}
                entities.update(d.entities)
                if d.name == "fmap/epi" and not entities.get("direction") and \
                        outcome.features is not None and outcome.features.pe_dir:
                    entities["direction"] = outcome.features.pe_dir
                entities = {**{k: v for k, v in entities.items() if k in ("subject", "session")},
                            **_allowed(entities, datatype, suffix)}
        label = outcome.verdict.entities.get("ceagent")
        if label and not d.excluded and new and not new.startswith("exclude/"):
            datatype, _, suffix = new.partition("/")
            withce = _allowed({**entities, "ceagent": label}, datatype, suffix)
            if "ceagent" in withce:
                entities = {**entities, "ceagent": label}
        exclude = d.excluded and not old.startswith("exclude/")
        renamed = exclude or (not d.excluded and (new != old or entities != before))
        changes.append(Change(index=int(key) if key.isdigit() else key, old=old, new=new,
                              entities=entities, exclude=exclude,
                              columns=columns(outcome), check=d.check, renamed=renamed))
    return changes


def apply_to_frame(df, changes: list[Change]):
    """Write ``changes`` into ``df`` (a copy): the classification columns,
    and, where the name changes, the datatype, suffix, entities and the
    rebuilt BIDS name. Returns ``(df, rows renamed)``."""
    import json

    from ...inventory.rebuild import rebuild_from_entities

    out = df.copy()
    renamed = 0
    for col in ("classification_confidence", "classification_alternatives",
                "classification_evidence"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(object)
    for ch in changes:
        for col, value in ch.columns.items():
            out.at[ch.index, col] = value
        if not ch.renamed:
            continue
        renamed += 1
        if ch.exclude:
            out.at[ch.index, "include"] = 0
            note = f"left out by the acquisition parameters ({ch.new.split('/', 1)[1]})"
            issues = str(out.at[ch.index, "issues"] or "") if "issues" in out.columns else ""
            out.at[ch.index, "issues"] = f"{issues}; {note}" if issues else note
            continue
        datatype, _, suffix = ch.new.partition("/")
        out.at[ch.index, "datatype"] = datatype
        out.at[ch.index, "bids_guess_datatype"] = datatype
        out.at[ch.index, "bids_guess_suffix"] = suffix
        out.at[ch.index, "entities"] = json.dumps(ch.entities, sort_keys=True)
    if renamed:
        out, _report = rebuild_from_entities(out)
    return out, renamed


__all__ += ["Change", "apply_to_frame", "rescore"]

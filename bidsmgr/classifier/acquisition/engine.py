"""Naming a series from how it was acquired: score, context, decision.

1. **Score each series alone** against every candidate of the rulebook
   (``rulebook.json``): each piece of evidence (a parameter, a word in the
   name) adds its weight, scaled by the source's weight in the settings.
   A softmax turns the scores into probabilities; the confidence is the top
   one.
2. **Look at the session as a whole** (:func:`context_pass`): an SBRef sits
   just before its run with the same geometry; a few-volume EPI phase-encoded
   opposite to a run is that run's fieldmap; two inversions of one 3-D
   gradient echo are an MP2RAGE; a phase series beside its magnitude twin is
   the same run's ``part-phase``. Context adds evidence and entities
   (``dir``, ``inv``, ``part``).
3. **Decide** (:func:`decide`), with dcm2niix's guess as an independent
   second opinion that this classifier never reads:

   * both agree: confirmed;
   * ours sure (at least ``apply_at``) and they differ: ours, saying what
     dcm2niix suggested;
   * ours unsure (between ``check_at`` and ``apply_at``): the one the name
     agrees with, marked to check;
   * ours silent (below ``check_at``): the other opinion, marked to check.

   A user's forced scan rule wins over all of it (it is applied before).

Every verdict keeps its evidence: the strongest lines for and against, so a
name can be explained and, when wrong, corrected with the setting that
caused it. Qt-free.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT, ClassifierConfig
from .features import SeriesFeatures

RULEBOOK = Path(__file__).with_name("rulebook.json")
#: The no-emit candidates: what the classifier says should not be converted.
EXCLUDE = "exclude"
_DIFFUSION_MAPS = {"FA", "ADC", "TRACEW", "TENSOR", "EXP", "EADC", "COLFA", "FA MAP",
                   "ADC MAP", "ISOTROPIC", "TRACE"}
_REFORMAT_TOKENS = {"MPR", "MIP", "MNIP", "PROJECTION IMAGE", "RESAMPLED", "MOCO", "CPR",
                    "CSA REPORT", "CSAPARALLEL", "CSA MPR"}
#: Maps a scanner computes from a perfusion series (ASL subtraction, CBF).
_PERFUSION_MAP_TOKENS = {"SUBTRACTION", "RCBF", "CBF", "CBV", "RCBV", "MTT", "TTP",
                         "PERFUSION_WEIGHTED", "PWI", "TTP_MAP"}


# ---------------------------------------------------------------------------
# The rulebook
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rulebook(delta: Optional[dict] = None) -> dict:
    """The shipped rulebook, with the user's ``delta`` (``{"candidates":
    {name: {...}}}``) laid over it: a candidate in the delta replaces the
    shipped one, ``{"name": null}`` removes it."""
    base = _load(str(RULEBOOK), RULEBOOK.stat().st_mtime)
    candidates = dict(base.get("candidates", {}))
    for name, entry in ((delta or {}).get("candidates") or {}).items():
        if entry is None:
            candidates.pop(name, None)
        else:
            candidates[name] = entry
    return {**base, "candidates": candidates}


# ---------------------------------------------------------------------------
# What the rulebook tests
# ---------------------------------------------------------------------------


def name_text(f: SeriesFeatures) -> str:
    """The series' names, lower case, separators as spaces, padded."""
    raw = f"{f.description} {f.protocol}".lower()
    return " " + re.sub(r"[^a-z0-9*]+", " ", raw).strip() + " "


def derived(f: SeriesFeatures, cfg: ClassifierConfig = DEFAULT) -> dict[str, Any]:
    """The facts the rulebook tests, derived from the features with the
    configured ranges. ``None`` where the series does not say."""
    a, s, g, e, c = cfg.anatomy, cfg.series, cfg.gradient_echo, cfg.echoes, cfg.contrast
    te, tr, ti = f.first_te, f.repetition_time, f.inversion_time
    n_echoes = len(f.echo_times)
    # Two echoes acquired, even when the image carries one echo time (a
    # phase difference is reconstructed from the pair).
    two_echoes = n_echoes == 2 or f.protocol_echoes == 2
    multi_echo = n_echoes >= e.multi_echo_min
    gre = f.is_gre and not f.is_epi
    seq = f.sequence_name.lower()
    # Siemens names every EPI "EP" in its scanning sequence: the sequence
    # name says whether it reads a gradient echo (epfid) or a spin echo
    # (epse, and the diffusion ep_b).
    epi_se = f.is_epi and (f.is_se and not f.is_gre or "epse" in seq
                           or bool(re.search(r"ep_?b\d", seq)))
    epi_gre = f.is_epi and not epi_se
    volumes = f.n_volumes or 0
    epi_run = f.is_epi and volumes >= cfg.series.run_min_volumes
    # Flow compensation: the scan option, or a Siemens 3-D FLASH whose name
    # says rephased (fl3d1r).
    flow_comp = "FC" in f.scan_options or bool(re.search(r"fl3d\d*r", seq))
    voxel = max(f.pixel_spacing or 99.0, f.slice_thickness or 99.0)
    se = f.is_se and not f.is_epi and not f.is_gre
    tokens = set(f.image_type)
    inverted = ti is not None and ti > 0
    known = te is not None and tr is not None

    return {
        "is_epi": f.is_epi,
        "is_derived": f.is_derived,
        "volumes": f.n_volumes,
        "n_echoes": n_echoes or None,
        "max_b": f.max_b if f.b_values else None,
        "n_directions": f.n_directions or None,
        "has_slice_timing": f.has_slice_timing,
        "part": f.part or None,
        "mt": f.mt,
        "n_orientations": f.n_orientations or None,
        "t1_ir_gre": gre and inverted and a.t1_ti_min <= ti <= a.t1_ti_max
        and f.acquisition_type == "3D" and not (tr is not None and tr >= a.mp2rage_tr_min),
        "t1_se": bool(se and known and te <= a.pd_te_max and tr < a.long_tr_min
                      and not inverted),
        "spoiled_3d": bool(gre and f.acquisition_type == "3D" and tr is not None
                           and tr < g.spoiled_3d_tr_max / 1000.0
                           and (f.flip_angle or 0) <= g.spoiled_3d_flip_max and n_echoes <= 1
                           and not inverted and f.part != "phase"),
        "flair_ir": inverted and ti >= a.flair_ti_min and not gre,
        "t2_se": bool(se and known and te >= a.t2_te_min and tr >= a.long_tr_min
                      and not (inverted and ti >= a.flair_ti_min) and not multi_echo),
        "pd_se": bool(se and known and te <= a.pd_te_max and tr >= a.long_tr_min
                      and not inverted),
        "gre_only": gre,
        "t2star_gre": bool(gre and te is not None and te >= a.t2star_te_min and n_echoes == 1
                           and not inverted),
        "multi_echo": multi_echo,
        "megre": gre and multi_echo,
        "mese": se and multi_echo,
        "mp2rage_part": bool(gre and inverted and f.acquisition_type == "3D"
                             and tr is not None and tr >= a.mp2rage_tr_min
                             and not f.is_derived),
        "uni_image": "UNI" in tokens,
        "t1_map_image": bool({"T1 MAP", "T1MAP"} & tokens),
        "fieldmap_echoes": bool(gre and n_echoes == 2 and f.part != "phase"
                                and (te or 99) < e.fieldmap_te_max),
        "fieldmap_phase": bool(gre and f.part == "phase" and n_echoes <= 2
                               and f.acquisition_type != "3D" or
                               (gre and f.part == "phase" and two_echoes)),
        "few_slices": bool((f.n_slices or 99) <= s.localizer_max_slices and not f.is_epi),
        "single_epi_volume": bool(f.is_epi and (f.n_volumes or 0) == 1),
        "epi_volumes": f.n_volumes if f.is_epi else None,
        "mt_short_te": bool(f.mt and gre and te is not None and te < g.mt_te_max),
        "b0_only": bool(f.is_epi and f.b_values and max(f.b_values) <= s.dwi_min_b
                        and "DIFFUSION" in tokens),
        "spoiled_2d_t1": bool(gre and f.acquisition_type != "3D" and te is not None
                              and te < g.spoiled_2d_te_max and tr is not None
                              and tr < g.spoiled_2d_tr_max
                              and (f.flip_angle or 0) >= g.spoiled_2d_flip_min
                              and not inverted and f.part != "phase"),
        # The echo time of a spin echo with a long repetition time (where T2
        # and proton density weighting part ways), else nothing.
        "se_long_tr_te": te if (se and tr is not None and tr >= a.long_tr_min
                                and not inverted and not multi_echo) else None,
        "field_in_hz": bool(gre and f.part == "phase" and n_echoes == 1 and not two_echoes),
        "reformat": bool(f.is_derived and tokens & _REFORMAT_TOKENS),
        "tof_angio": bool(gre and f.acquisition_type == "3D" and flow_comp
                          and tr is not None and tr * 1000.0 <= g.angio_tr_max
                          and te is not None and te <= g.angio_te_max
                          and g.angio_flip_min <= (f.flip_angle or 0) <= g.angio_flip_max
                          and not inverted and f.part != "phase"),
        "mp_gre": bool(gre and f.acquisition_type == "3D" and "MP" in f.sequence_variant),
        "turbo_spin_echo": bool(se and (f.echo_train_length or 0) >= a.tse_etl_min),
        "high_res_3d": bool(f.acquisition_type == "3D" and voxel <= a.high_res_max_mm),
        "bold_te": te if (epi_gre and epi_run) else None,
        "epi_run_tr": tr if epi_run else None,
        "contrast_agent": bool(f.contrast_agent),
        "contrast_bolus": bool(epi_gre and f.contrast_agent and volumes >= c.dsc_min_volumes
                               and not (f.b_values and f.max_b > s.dwi_min_b)),
        "perfusion_image": "PERFUSION" in tokens,
        "asl_labelled": (f.asl_mode or 0) >= 2,
        # Siemens marks every image of an ASL sequence ASL, its M0 too.
        "asl_image": "ASL" in tokens and not f.is_derived and volumes > c.m0_max_volumes,
        "asl_reference": bool((f.is_epi or "tgse" in seq) and 0 < volumes <= c.m0_max_volumes
                              and ((f.asl_mode or 0) >= 2 or "ASL" in tokens or "tgse" in seq)),
        "asl_series": bool((f.asl_mode or 0) >= 2 and volumes > c.m0_max_volumes),
        "perfusion_map": bool(f.is_derived and tokens & _PERFUSION_MAP_TOKENS),
        "_run_min": s.run_min_volumes,
    }


_SOURCES = ("parameters", "names", "context")


def rulebook_problems(book: dict) -> list[str]:
    """What is wrong with a rulebook (or a change to one), in sentences; an
    empty list when it can be used. Every parameter line must test a fact
    :func:`derived` knows, every name line a pattern that compiles."""
    known = set(derived(SeriesFeatures())) | {"volumes", "n_echoes", "max_b",
                                               "n_directions", "part"}
    problems: list[str] = []
    candidates = book.get("candidates")
    if not isinstance(candidates, dict):
        return ["it has no \"candidates\""]
    for name, entry in candidates.items():
        if entry is None:
            continue
        if "/" not in name:
            problems.append(f"{name}: a name is datatype/suffix (or exclude/<reason>)")
        if not isinstance(entry, dict) or not isinstance(entry.get("evidence"), list):
            problems.append(f"{name}: no \"evidence\" list")
            continue
        for n, item in enumerate(entry["evidence"], start=1):
            where = f"{name}, evidence {n}"
            if not isinstance(item, dict):
                problems.append(f"{where}: not an object")
                continue
            if item.get("source") not in _SOURCES:
                problems.append(f"{where}: source is one of {', '.join(_SOURCES)}")
            try:
                float(item.get("weight"))
            except (TypeError, ValueError):
                problems.append(f"{where}: the weight is not a number")
            if "pattern" in item:
                try:
                    re.compile(str(item["pattern"]))
                except re.error as exc:
                    problems.append(f"{where}: the pattern does not compile ({exc})")
            elif item.get("feature") not in known:
                problems.append(f"{where}: no feature named {item.get('feature')!r}")
    return problems


def rulebook_delta(book: dict, *, full: bool) -> dict:
    """The change ``book`` makes to the shipped rulebook: the candidates it
    adds or alters, and (``full``: it is a whole rulebook, not a change)
    the shipped ones it leaves out, as ``None``."""
    shipped = _load(str(RULEBOOK), RULEBOOK.stat().st_mtime).get("candidates", {})
    given = book.get("candidates") or {}
    delta = {name: entry for name, entry in given.items() if shipped.get(name) != entry}
    if full:
        delta.update({name: None for name in shipped if name not in given})
    return {"candidates": delta} if delta else {}


def _resolve(value, cfg: ClassifierConfig):
    if isinstance(value, str) and value.startswith("cfg:"):
        section, _, name = value[4:].partition(".")
        return getattr(getattr(cfg, section), name)
    return value


def _support(item: dict, facts: dict, text: str, cfg: ClassifierConfig) -> tuple[float, Any]:
    """``(support in [-1, 1], the value it looked at)``; 0 when the series
    does not say."""
    if "pattern" in item:
        return (1.0, None) if re.search(item["pattern"], text) else (0.0, None)
    value = facts.get(item.get("feature", ""))
    if value is None:
        return 0.0, None
    otherwise = float(item.get("else", 0))
    if "is" in item:
        return (1.0 if value == item["is"] else otherwise), value
    soft = float(item.get("soft", 0) or 0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, value
    if "min" in item:
        lo = float(_resolve(item["min"], cfg))
        if number >= lo:
            return 1.0, value
        if soft and number >= lo - soft:
            return (number - (lo - soft)) / soft, value
        return otherwise, value
    if "max" in item:
        hi = float(_resolve(item["max"], cfg))
        if number <= hi:
            return 1.0, value
        if soft and number <= hi + soft:
            return (hi + soft - number) / soft, value
        return otherwise, value
    if "between" in item:
        lo, hi = (float(_resolve(v, cfg)) for v in item["between"])
        return (1.0 if lo <= number <= hi else otherwise), value
    return 0.0, value


# ---------------------------------------------------------------------------
# Scoring one series
# ---------------------------------------------------------------------------


@dataclass
class Line:
    """One piece of evidence as it counted."""

    candidate: str
    source: str            # "parameters" | "names" | "context"
    text: str
    points: float

    def to_dict(self) -> dict:
        return {"for": self.candidate, "source": self.source, "text": self.text,
                "points": round(self.points, 2)}


@dataclass
class Verdict:
    """What the classifier makes of one series."""

    scores: dict[str, float] = field(default_factory=dict)
    lines: list[Line] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    #: Why an entity was set ({"ceagent": "the header names ..."}).
    entity_why: dict[str, str] = field(default_factory=dict)
    #: An entity rests on the name alone: the row is marked to check.
    entity_check: bool = False

    def add(self, candidate: str, source: str, text: str, points: float) -> None:
        if not points:
            return
        self.scores[candidate] = self.scores.get(candidate, 0.0) + points
        self.lines.append(Line(candidate, source, text, points))

    def finish(self) -> "Verdict":
        if not self.scores:
            self.probabilities = {}
            return self
        top = max(self.scores.values())
        exps = {c: math.exp(s - top) for c, s in self.scores.items()}
        total = sum(exps.values())
        self.probabilities = {c: e / total for c, e in
                              sorted(exps.items(), key=lambda kv: -kv[1])}
        return self

    @property
    def best(self) -> str:
        return next(iter(self.probabilities), "")

    @property
    def confidence(self) -> float:
        return next(iter(self.probabilities.values()), 0.0)

    def alternatives(self, n: int = 2) -> list[tuple[str, float]]:
        return list(self.probabilities.items())[1:1 + n]

    def evidence_for(self, candidate: str, n: int = 6) -> list[Line]:
        mine = [ln for ln in self.lines if ln.candidate == candidate]
        return sorted(mine, key=lambda ln: -abs(ln.points))[:n]


def in_domain(f: SeriesFeatures) -> bool:
    """Whether ``f`` is an MR image series, the only kind this classifier
    names. A spectroscopy series has a 1 x 1 "image" and an echo and
    repetition time like any spin echo, and was named a PD-weighted image."""
    if f.is_spectroscopy or f.dicom_modality not in ("", "MR"):
        return False
    return not (f.rows is None and f.n_volumes is None)


def domain_datatypes(book: Optional[dict] = None) -> frozenset[str]:
    """The datatypes the rulebook can name (``exclude`` aside). A name in
    any other datatype (``mrs``, ``pet``, ...) came from a rule that knows
    more about that series than the parameters do, and is never replaced."""
    book = book or rulebook()
    return frozenset(n.split("/", 1)[0] for n in book["candidates"]) - {EXCLUDE}


def score(f: SeriesFeatures, cfg: ClassifierConfig = DEFAULT,
          book: Optional[dict] = None) -> Verdict:
    """Every candidate's score for ``f`` alone (before context)."""
    book = book or rulebook(cfg.rulebook)
    verdict = Verdict()
    if not in_domain(f):
        # Not an MR image (a physiological log, a report, spectroscopy, a
        # PET or CT series): not ours to name; the scan's own rules handle
        # them, and they ran on the header before us.
        return verdict
    if f.is_derived and set(f.image_type) & _DIFFUSION_MAPS:
        # A map the scanner computed from a diffusion series (FA, ADC, trace,
        # tensor): named by its own rules, which know BIDS's suffixes for them.
        return verdict
    facts = derived(f, cfg)
    text = name_text(f)
    scale = {"parameters": cfg.weights.parameters, "names": cfg.weights.names,
             "context": cfg.weights.context}
    dsc_as = "func/cbv" if cfg.contrast.dsc == "cbv" else "exclude/dsc"
    for candidate, entry in book["candidates"].items():
        if candidate == "exclude/dsc":
            candidate = dsc_as
        verdict.scores.setdefault(candidate, float(entry.get("prior", 0.0)))
        for item in entry.get("evidence", []):
            support, value = _support(item, facts, text, cfg)
            if not support:
                continue
            points = float(item.get("weight", 0)) * support * scale.get(item.get("source"), 1.0)
            why = str(item.get("why", ""))
            if value is not None and "{value}" in why:
                shown = f"{value:g}" if isinstance(value, float) else str(value)
                why = why.replace("{value}", shown)
            if support < 0:
                why = f"not {why}" if not why.startswith("not ") else why[4:]
            verdict.add(candidate, item.get("source", "parameters"), why, points)
    label, why, from_header = contrast_label(f, cfg)
    if label:
        verdict.entities["ceagent"] = label
        verdict.entity_why["ceagent"] = why
        verdict.entity_check = verdict.entity_check or not from_header
    return verdict


#: A series name saying "after contrast" (post Gd, +C, KM, Kontrastmittel).
#: Deliberately not a bare "post": a session can be called post.
_CONTRAST_NAME = re.compile(
    r"\bpost ?(gd|gad\w*|contrast|km|ce|c|kontrast\w*)\b|\+ ?(c|gd|km)\b|\bgd\b|"
    r"\bgad\b|gadolin\w*|\bgado\w+|kontrast\w*|\bkm\b|\bcontrast\b|\bce\b")


def _label(text: str) -> str:
    """A BIDS label from an agent's name: amounts and units dropped, letters
    and digits kept ("24ML MULTIHANCE" -> "multihance")."""
    text = re.sub(r"(?i)\b\d+([.,]\d+)?\s*(ml|cc|mmol|mg|%|mmol/kg)?\b", " ", text)
    return re.sub(r"[^0-9a-zA-Z]", "", text).lower()[:24]


def contrast_label(f: SeriesFeatures, cfg: ClassifierConfig = DEFAULT) -> tuple[str, str, bool]:
    """``(ce label, why, whether the header said so)``; ``("", "", False)``
    for a series acquired without a contrast agent (or with the entity
    turned off)."""
    c = cfg.contrast
    if c.entity == "none":
        return "", "", False
    agent = f.contrast_agent
    named = c.from_names and bool(_CONTRAST_NAME.search(name_text(f)))
    if not agent and not named:
        return "", "", False
    label = (_label(c.fixed_label) if c.entity == "fixed" or not agent else _label(agent)) \
        or _label(c.fixed_label)
    if agent:
        return label, f"the header names a contrast agent ({agent})", True
    return label, "the name says it was acquired after a contrast agent", False


# ---------------------------------------------------------------------------
# The session as a whole
# ---------------------------------------------------------------------------


@dataclass
class Series:
    """One series of a session, as the context pass sees it."""

    key: str                      # the caller's id for it
    features: SeriesFeatures
    order: float                  # acquisition order (seconds, or index)
    verdict: Verdict = field(default_factory=Verdict)


def _geometry(f: SeriesFeatures) -> tuple:
    return (f.rows, f.columns, f.n_slices)


def _opposite(a: str, b: str) -> bool:
    pairs = {("AP", "PA"), ("PA", "AP"), ("LR", "RL"), ("RL", "LR"), ("IS", "SI"),
             ("SI", "IS")}
    return (a, b) in pairs


def _stem(text: str) -> str:
    text = re.sub(r"(?i)(_?(inv|inv-)?[12]|_?uni.*|_?t1.?images?|_?t1.?map.*|_?sbref|"
                  r"_?(ph|phase|mag))$", "", text.strip())
    return text.lower()


def context_pass(session: list[Series], cfg: ClassifierConfig = DEFAULT) -> None:
    """Context evidence and entities across one session (in place)."""
    w = cfg.weights.context
    s = cfg.series
    # A series the classifier abstained on (no image, a scanner's map) stays
    # out of it: context adds evidence, it does not start an opinion.
    ordered = sorted((x for x in session if x.verdict.scores), key=lambda x: x.order)
    runs = [x for x in ordered if x.verdict.best in ("func/bold", "dwi/dwi")
            and (x.features.n_volumes or 0) >= s.run_min_volumes]
    for i, x in enumerate(ordered):
        f = x.features
        # An SBRef: one EPI volume just before a multiband run of the same
        # geometry (a single-band sequence has no SBRef).
        if f.is_epi and (f.n_volumes or 0) == 1:
            later = [y for y in ordered[i + 1:i + 1 + s.sbref_lookahead] if y in runs
                     and _geometry(y.features) == _geometry(f)
                     and ((y.features.multiband or 1) > 1
                          or "sbref" in f.description.lower())]
            if later:
                kind = later[0].verdict.best.split("/")[0]
                x.verdict.add(f"{kind}/sbref", "context",
                              f"one volume just before a {later[0].verdict.best} run with "
                              "the same geometry", 3.0 * w)
        # A reversed-phase reference: few EPI volumes, opposite to a run,
        # or to the other reference of a pair. Without an opposite partner
        # (or with no direction known, which GE does not record) it corrects
        # nothing, and is far more likely a short run or a test.
        if f.is_epi and 0 < (f.n_volumes or 0) <= s.reference_max_volumes:
            partner = min((y for y in runs if f.pe_dir and _geometry(y.features) == _geometry(f)
                           and _opposite(f.pe_dir, y.features.pe_dir)),
                          key=lambda y: abs(y.order - x.order), default=None)
            pair = None if partner is not None or not f.pe_dir else next(
                (y for y in ordered if y is not x and y.features.is_epi
                 and 0 < (y.features.n_volumes or 0) <= s.reference_max_volumes
                 and _geometry(y.features) == _geometry(f)
                 and _opposite(f.pe_dir, y.features.pe_dir)), None)
            if partner is not None:
                x.verdict.add("fmap/epi", "context",
                              f"phase-encoded {f.pe_dir}, opposite to a run "
                              f"({partner.features.pe_dir}) with the same geometry", 4.0 * w)
                x.verdict.entities["direction"] = f.pe_dir
            elif pair is not None:
                x.verdict.add("fmap/epi", "context",
                              f"phase-encoded {f.pe_dir}, opposite to another reference "
                              f"({pair.features.pe_dir}): a pair", 3.0 * w)
                x.verdict.entities["direction"] = f.pe_dir
            else:
                x.verdict.add("fmap/epi", "context",
                              "nothing with its geometry is phase-encoded the other way"
                              if f.pe_dir else "its phase-encoding direction is not recorded",
                              -3.0 * w)
        # Magnitude and phase twins of one acquisition: the same run, two parts.
        if f.part in ("phase", "mag"):
            twin = next((y for y in ordered if y is not x
                         and y.features.part in ("phase", "mag")
                         and y.features.part != f.part
                         and y.features.description == f.description
                         and _geometry(y.features) == _geometry(f)
                         and y.features.n_volumes == f.n_volumes), None)
            if twin is not None:
                x.verdict.entities["part"] = f.part
    # MP2RAGE: two inversions of one sequence, numbered by inversion time.
    inversions = [x for x in ordered if derived(x.features, cfg).get("mp2rage_part")]
    groups: dict[tuple, list[Series]] = {}
    for x in inversions:
        groups.setdefault((x.features.sequence_name, _geometry(x.features),
                           _stem(x.features.description)), []).append(x)
    for members in groups.values():
        tis = sorted({m.features.inversion_time for m in members})
        if len(tis) < 2:
            continue
        for m in members:
            m.verdict.add("anat/MP2RAGE", "context",
                          f"one of {len(tis)} inversions of the same 3-D gradient echo", 3.0 * w)
            m.verdict.entities["inversion"] = str(tis.index(m.features.inversion_time) + 1)
    for x in session:
        x.verdict.finish()


def classify_session(session: list[Series], cfg: ClassifierConfig = DEFAULT,
                     book: Optional[dict] = None) -> None:
    """Score every series of a session and apply the context (in place)."""
    book = book or rulebook(cfg.rulebook)
    for x in session:
        x.verdict = score(x.features, cfg, book).finish()
    context_pass(session, cfg)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

#: Suffixes one dcm2niix series splits into, compared as one.
#: A GE or Philips B0 map is one series too: dcm2niix writes its magnitude
#: and its field map in Hz (``_fieldmaphz``) side by side.
_FAMILY = {"magnitude": "fmap/gre", "magnitude1": "fmap/gre", "magnitude2": "fmap/gre",
           "phasediff": "fmap/gre", "phase1": "fmap/gre", "phase2": "fmap/gre",
           "fieldmap": "fmap/gre"}


def same(a: str, b: str) -> bool:
    """Whether two ``datatype/suffix`` names agree (a fieldmap's magnitude
    and phase difference are one series' outputs)."""
    if not a or not b:
        return False
    if a == b:
        return True
    da, _, sa = a.partition("/")
    db, _, sb = b.partition("/")
    return da == db == "fmap" and _FAMILY.get(sa) == _FAMILY.get(sb) is not None


@dataclass
class Decision:
    """The name a series gets, and why."""

    name: str                     # "func/bold", "exclude/localizer", "" (none)
    decided_by: str               # "agreement" | "parameters" | "names" | "dcm2niix" | "rules" | ""
    check: bool
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)
    ours: str = ""
    other: str = ""
    note: str = ""

    @property
    def excluded(self) -> bool:
        return self.name.startswith(EXCLUDE + "/")


def decide(verdict: Verdict, other: str, names_say: str, cfg: ClassifierConfig = DEFAULT,
           *, other_by: str = "dcm2niix") -> Decision:
    """Combine ours (``verdict``) with the other opinion (``other``, e.g.
    dcm2niix's ``func/bold``; ``""`` when it had none) and what the name
    alone says (``names_say``)."""
    d = _decide(verdict, other, names_say, cfg, other_by=other_by)
    if d.name == "func/cbv" and not d.entities.get("task"):
        d.entities["task"] = "dsc"
    if verdict.entity_check and not d.excluded:
        d.check = True
        d.note = (d.note + "; " if d.note else "") + verdict.entity_why.get("ceagent", "")
    return d


def _decide(verdict: Verdict, other: str, names_say: str, cfg: ClassifierConfig,
            *, other_by: str) -> Decision:
    ours, conf = verdict.best, verdict.confidence
    g = cfg.general
    if not ours or conf < g.check_at:
        if other:
            return Decision(other, other_by, True, conf, {}, ours, other,
                            "the parameters do not decide")
        return Decision("", "", True, conf, {}, ours, other, "no opinion")
    if same(ours, other):
        return Decision(other if other.startswith("fmap/") else ours, "agreement", False,
                        max(conf, 0.9), dict(verdict.entities), ours, other)
    if conf >= g.apply_at:
        note = f"{other_by} suggests {other}" if other else ""
        return Decision(ours, "parameters", False, conf, dict(verdict.entities), ours, other,
                        note)
    # Unsure: the opinion the name agrees with, marked to check.
    if other and same(names_say, other) and not same(names_say, ours):
        return Decision(other, other_by, True, conf, {}, ours, other,
                        f"the name agrees with {other_by}")
    if not other and names_say and not same(names_say, ours):
        return Decision(names_say, "names", True, conf, {}, ours, other,
                        "the parameters are unsure and the name says otherwise")
    return Decision(ours, "parameters", True, conf, dict(verdict.entities), ours, other,
                    f"{other_by} suggests {other}" if other else "")


def names_only(f: SeriesFeatures, cfg: ClassifierConfig = DEFAULT,
               book: Optional[dict] = None) -> str:
    """What the name alone says (the best candidate on name evidence)."""
    book = book or rulebook(cfg.rulebook)
    text = name_text(f)
    best, top = "", 0.0
    for candidate, entry in book["candidates"].items():
        points = sum(float(i.get("weight", 0)) for i in entry.get("evidence", [])
                     if i.get("source") == "names" and re.search(i["pattern"], text))
        if points > top:
            best, top = candidate, points
    return best


def report(verdict: Verdict, decision: Decision) -> dict:
    """What is kept with the row: confidence, alternatives, evidence."""
    chosen = decision.name or verdict.best
    return {
        "confidence": round(verdict.confidence, 3),
        "alternatives": [{"name": c, "p": round(p, 3)} for c, p in verdict.alternatives(3)],
        "evidence": {
            "chosen": chosen,
            "ours": decision.ours,
            "other": decision.other,
            "decided_by": decision.decided_by,
            "check": decision.check,
            "note": decision.note,
            "entities": dict(verdict.entity_why),
            "for": [ln.to_dict() for ln in verdict.evidence_for(chosen)],
            "runner_up": [ln.to_dict() for ln in verdict.evidence_for(
                next((c for c, _p in verdict.alternatives(1)), ""), 3)],
        },
    }


__all__ = ["Decision", "EXCLUDE", "Line", "Series", "Verdict", "classify_session",
           "context_pass", "contrast_label", "decide", "derived", "domain_datatypes",
           "in_domain", "name_text", "names_only", "report", "rulebook", "rulebook_delta",
           "rulebook_problems", "same", "score"]

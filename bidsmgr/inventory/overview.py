"""What the inventory will write, compared across subjects.

The inspection table answers questions about one row. The questions a curator
has about the whole table are about its SHAPE: does every subject have the same
recordings, was one task spelled two ways, did one visit end up as five
sessions. Those are differences BETWEEN rows, and a table of two thousand rows
does not show them.

So this counts what each subject (and session) will have, kind by kind, and
says where one differs from the others it should resemble. Measured on real
inventories before it was written: a TCIA visit whose series each carry their
own study, so one visit became five sessions; an EEG folder with 1,861 task
labels for 1,903 recordings, because the labels hold file names; a dataset
whose second subject has no session while the first has two.

Two rules, the same as the Editor's dashboard (:mod:`bidsmgr.editor.dashboard`):

**Every figure is one a user could count by hand.** A cell is a number of
recordings; a finding names the subjects and the kind.

**Nothing is called unusual without enough to compare.** A count is usual when
most of the comparable subjects share it (:data:`USUAL_SHARE`), and there must
be at least :data:`MIN_PEERS` of them. Two subjects that differ are just two
subjects; there is no telling which one is wrong.

Qt-free and pure data: the GUI computes it on a worker thread from a snapshot.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Column levels, most to least specific.
LEVELS = ("labels", "task", "suffix")
#: What each level is called where a user chooses it.
LEVEL_TITLES = {
    "auto": "Automatic",
    "labels": "Every label",
    "task": "Task only",
    "suffix": "Datatype and suffix",
}

#: The automatic level is the most specific one with at most this many
#: columns: past it a grid is read by scrolling, not at a glance.
MAX_AUTO_COLUMNS = 40
#: A count is what is USUAL when at least this share of the comparable
#: subjects have it.
USUAL_SHARE = 0.6
#: With fewer comparable subjects than this, nothing is usual.
MIN_PEERS = 3
#: A recording the others lack is named when at most this share of them has
#: it; a recording a third of the subjects have is a protocol with a branch,
#: not something to look at.
RARE_SHARE = 0.2
#: ... and only among at least this many: "only one of five subjects has it"
#: is noise more often than news (a corpus of unrelated scans, a split visit).
RARE_MIN_PEERS = 6
#: Labels look like identifiers when there are at least this many different
#: ones and they number at least this share of the recordings carrying them.
ID_LIKE_MIN = 10
ID_LIKE_SHARE = 0.5
#: Two label values at least this similar (difflib's ratio) read as one value
#: spelled two ways.
SIMILAR_RATIO = 0.75
#: How many names a finding lists before saying "and N more".
NAMES_LISTED = 4
#: More subjects than this with several sessions on one day are one finding.
SAME_DAY_LISTED = 3

#: Entities that do not make a different KIND of recording: the subject and
#: session are the grid's rows, and runs are what a cell counts.
_ROW_ENTITIES = frozenset({"sub", "ses", "run"})
_DATATYPE_ORDER = ("anat", "func", "fmap", "dwi", "perf", "mrs", "pet", "eeg",
                   "meg", "ieeg", "nirs", "motion", "beh", "micr")
_SEVERITY_ORDER = {"err": 0, "warn": 1, "info": 2}
#: How a modality is written for a reader.
_MODALITY_NAMES = {"mri": "MRI", "eeg": "EEG", "meg": "MEG", "pet": "PET",
                   "nirs": "NIRS", "eyetrack": "eye tracking"}
#: What a label of each entity stands for, in a sentence.
_ENTITY_MEANING = {"task": "tasks", "acq": "acquisitions", "rec": "reconstructions",
                   "dir": "phase directions", "ce": "contrast agents"}


# ---------------------------------------------------------------------------
# What it returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """One kind of recording: a column of the grid."""

    datatype: str
    suffix: str
    labels: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """``func/task-rest_bold``: the tail of a BIDS name, under its folder."""
        return f"{self.datatype}/{'_'.join((*self.labels, self.suffix))}"

    @property
    def caption(self) -> str:
        """The labels alone (``task-rest acq-mb``), empty for none."""
        return " ".join(self.labels)


@dataclass(frozen=True)
class Visit:
    """One subject, or one session of a subject: a row of the grid."""

    subject: str
    session: str = ""

    @property
    def label(self) -> str:
        return f"{self.subject} {self.session}".strip()


@dataclass(frozen=True)
class Cell:
    """What one visit has of one kind."""

    #: The recordings to convert (inventory row positions).
    rows: tuple[int, ...] = ()
    #: Recordings of this kind that are left out of the conversion.
    left_out: tuple[int, ...] = ()
    #: What most comparable visits have, ``None`` when nothing is usual.
    usual: Optional[int] = None
    #: How many visits this one was compared with, itself included.
    peers: int = 0
    #: How the count differs: "", "missing", "fewer", "extra" or "rare".
    state: str = ""

    @property
    def count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class Finding:
    """One thing worth a look, and the rows that show it."""

    key: str
    severity: str          # "err", "warn" or "info"
    text: str
    rows: tuple[int, ...] = ()


@dataclass
class Overview:
    """The inventory, counted by subject and kind of recording."""

    level: str = "suffix"
    #: Columns each level would have.
    level_columns: dict[str, int] = field(default_factory=dict)
    #: Entities whose labels look like identifiers (``{"task": 1861}``).
    id_like: dict[str, int] = field(default_factory=dict)
    visits: list[Visit] = field(default_factory=list)
    kinds: list[Kind] = field(default_factory=list)
    #: Every row of each visit, the ones left out too.
    visit_rows: list[tuple[int, ...]] = field(default_factory=list)
    #: The recordings to convert of each kind.
    kind_rows: list[tuple[int, ...]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    subjects: int = 0
    sessions: int = 0
    recordings: int = 0
    left_out: int = 0
    datatypes: int = 0
    by_modality: dict[str, int] = field(default_factory=dict)
    #: Whether any visit had enough others to be compared with.
    compared: bool = False
    _cells: dict[tuple[int, int], Cell] = field(default_factory=dict, repr=False)

    def cell(self, visit: int, kind: int) -> Cell:
        return self._cells.get((visit, kind), _EMPTY)

    def modality_line(self) -> str:
        """``30 MRI, 10 EEG`` by count, largest first."""
        parts = sorted(self.by_modality.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{n:,} {_MODALITY_NAMES.get(m, m.upper())}" for m, n in parts)


_EMPTY = Cell()


# ---------------------------------------------------------------------------
# Reading the rows
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    pos: int
    included: bool
    subject: str
    session: str
    datatype: str
    suffix: str
    pairs: tuple[tuple[str, str], ...]
    name: str
    dataset: str
    modality: str
    day: str
    check: bool


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:   # NaN
        return ""
    return str(value).strip()


def _split_name(name: str) -> tuple[tuple[tuple[str, str], ...], str]:
    """``sub-01_task-rest_bold`` into its entity pairs and its suffix."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].split(".", 1)[0]
    tokens = [t for t in base.split("_") if t]
    if not tokens:
        return (), ""
    pairs = []
    for token in tokens[:-1]:
        key, sep, value = token.partition("-")
        if sep:
            pairs.append((key, value))
    return tuple(pairs), tokens[-1]


def _day(study_date: str, recording_time: str) -> str:
    """The calendar day a row was recorded, ``YYYY-MM-DD`` or ""."""
    digits = re.sub(r"\D", "", study_date)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    match = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", recording_time)
    return "-".join(match.groups()) if match else ""


def _marked_to_check(evidence: str) -> bool:
    if not evidence or "check" not in evidence:
        return False
    try:
        return bool(json.loads(evidence).get("check"))
    except (ValueError, AttributeError):
        return False


def _conversion_includes(value) -> bool:
    """The conversion's own reading of the include cell: exactly ``1``."""
    return _text(value) == "1"


def _rows(df, included: Optional[Sequence[bool]]) -> list[_Row]:
    n = len(df)

    def column(name: str) -> list:
        return df[name].tolist() if name in df.columns else [""] * n

    include = column("include")
    if included is None:
        flags = ([_conversion_includes(v) for v in include] if "include" in df.columns
                 else [True] * n)
    else:
        flags = [bool(v) for v in included]
    names, subjects, sessions = column("bids_name"), column("participant_id"), column("session")
    datatypes, datasets, modalities = column("datatype"), column("dataset"), column("modality")
    study_dates, recording_times = column("study_date"), column("recording_time")
    evidence = column("classification_evidence")
    out = []
    for pos in range(n):
        name = _text(names[pos])
        pairs, suffix = _split_name(name) if name else ((), "")
        named = dict(pairs)
        subject = _text(subjects[pos]) or (f"sub-{named['sub']}" if "sub" in named else "")
        session = _text(sessions[pos]) or (f"ses-{named['ses']}" if "ses" in named else "")
        if session and not session.startswith("ses-"):
            session = f"ses-{session}"
        out.append(_Row(
            pos=pos, included=flags[pos] if pos < len(flags) else True,
            subject=subject or "(no subject)", session=session,
            datatype=_text(datatypes[pos]), suffix=suffix, pairs=pairs, name=name,
            dataset=_text(datasets[pos]), modality=_text(modalities[pos]).lower(),
            day=_day(_text(study_dates[pos]), _text(recording_times[pos])),
            check=_marked_to_check(_text(evidence[pos])),
        ))
    return out


def _kind(row: _Row, level: str) -> Kind:
    if level == "suffix":
        labels: tuple[str, ...] = ()
    elif level == "task":
        labels = tuple(f"task-{v}" for k, v in row.pairs if k == "task")
    else:
        labels = tuple(f"{k}-{v}" for k, v in row.pairs if k not in _ROW_ENTITIES)
    return Kind(row.datatype, row.suffix, labels)


def _natural(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", text))


def _kind_order(kind: Kind) -> tuple:
    try:
        rank = _DATATYPE_ORDER.index(kind.datatype)
    except ValueError:
        rank = len(_DATATYPE_ORDER)
    return (rank, kind.datatype, _natural(kind.caption), kind.suffix)


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``, ``a, b, c, d and 3 more``."""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    if len(names) > NAMES_LISTED + 1:
        return f"{', '.join(names[:NAMES_LISTED])} and {len(names) - NAMES_LISTED} more"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _amount(count: int) -> str:
    return "one" if count == 1 else f"{count:,}"


def _noun(visits: Sequence[Visit], count: int) -> str:
    word = "session" if any(v.session for v in visits) else "subject"
    return word if count == 1 else f"{word}s"


# ---------------------------------------------------------------------------
# Recordings of one kind across subjects (the Filter panel's Recordings mode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingGroup:
    """Every recording of one kind, in every subject, left out or not."""

    datatype: str          # "" for series with no BIDS name yet
    title: str             # "task-rest_bold"; a series description when not named
    rows: tuple[int, ...]  # inventory row positions
    subjects: int


def recording_groups(df, *, level: str = "labels") -> list[RecordingGroup]:
    """The table's recordings grouped by kind (as the Overview's columns at
    ``level``) across subjects, the rows left out INCLUDED, so a kind can be
    ticked or unticked everywhere at once. A series with no name yet (a
    localizer, a scout) is grouped by its description instead."""
    rows = _rows(df, [True] * len(df))
    groups: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for r in rows:
        if r.name and r.datatype:
            kind = _kind(r, level if level in LEVELS else "labels")
            datatype = "derivatives" if kind.datatype.startswith("derivatives") else kind.datatype
            groups[(datatype, "_".join((*kind.labels, kind.suffix)))].append(r)
        else:
            description = _text(df.iloc[r.pos].get("sequence", "")) if "sequence" in df.columns else ""
            groups[("", description or "(no description)")].append(r)

    def order(key):
        datatype, title = key
        try:
            rank = _DATATYPE_ORDER.index(datatype)
        except ValueError:
            rank = len(_DATATYPE_ORDER) + (1 if not datatype else 0)
        return (rank, datatype, _natural(title))

    return [RecordingGroup(datatype=k[0], title=k[1], rows=tuple(r.pos for r in members),
                           subjects=len({r.subject for r in members}))
            for k, members in sorted(groups.items(), key=lambda kv: order(kv[0]))]


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def _id_like(rows: list[_Row]) -> dict[str, int]:
    """Entities whose labels differ for nearly every recording carrying them."""
    values: dict[str, set] = defaultdict(set)
    carried: Counter = Counter()
    for row in rows:
        for key, value in row.pairs:
            if key in _ROW_ENTITIES:
                continue
            values[key].add(value)
            carried[key] += 1
    return {key: len(seen) for key, seen in values.items()
            if len(seen) >= ID_LIKE_MIN and len(seen) >= ID_LIKE_SHARE * carried[key]}


def _choose_level(level: str, columns: dict[str, int], id_like: dict[str, int]) -> str:
    if level in LEVELS:
        return level
    if columns.get("labels", 0) <= MAX_AUTO_COLUMNS and not id_like:
        return "labels"
    if columns.get("task", 0) <= MAX_AUTO_COLUMNS and "task" not in id_like:
        return "task"
    return "suffix"


def build_overview(df, included: Optional[Sequence[bool]] = None, *,
                   level: str = "auto") -> Overview:
    """Count ``df`` (an inventory) by subject and kind of recording.

    ``included`` gives each row's include flag in row order; without it the
    conversion's own rule applies (the cell is exactly ``1``). ``level`` is
    one of :data:`LEVELS` or ``"auto"``.
    """
    rows = _rows(df, included)
    named = [r for r in rows if r.name and r.datatype]
    converting = [r for r in named if r.included]

    overview = Overview()
    overview.recordings = sum(1 for r in rows if r.included)
    overview.left_out = sum(1 for r in rows if not r.included)
    overview.by_modality = dict(Counter(r.modality or "other" for r in rows if r.included))
    overview.datatypes = len({r.datatype for r in converting})
    overview.subjects = len({r.subject for r in converting})
    overview.sessions = len({(r.subject, r.session) for r in converting if r.session})

    overview.id_like = _id_like(converting)
    overview.level_columns = {lv: len({_kind(r, lv) for r in converting}) for lv in LEVELS}
    overview.level = _choose_level(level, overview.level_columns, overview.id_like)

    # Rows and columns.
    visit_index: dict[Visit, list[int]] = defaultdict(list)
    modality_of: dict[int, str] = {}
    for r in named:
        visit_index[Visit(r.subject, r.session)].append(r.pos)
        modality_of[r.pos] = r.modality or "other"
    overview.visits = sorted(visit_index, key=lambda v: (_natural(v.subject), _natural(v.session)))
    overview.visit_rows = [tuple(visit_index[v]) for v in overview.visits]
    kinds = {_kind(r, overview.level) for r in converting}
    overview.kinds = sorted(kinds, key=_kind_order)
    v_of = {v: i for i, v in enumerate(overview.visits)}
    k_of = {k: i for i, k in enumerate(overview.kinds)}
    kind_rows: dict[int, list[int]] = defaultdict(list)
    taken: dict[tuple[int, int], list[int]] = defaultdict(list)
    dropped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in named:
        kind = _kind(r, overview.level)
        if kind not in k_of:
            continue
        at = (v_of[Visit(r.subject, r.session)], k_of[kind])
        (taken if r.included else dropped)[at].append(r.pos)
        if r.included:
            kind_rows[at[1]].append(r.pos)
    overview.kind_rows = [tuple(kind_rows[i]) for i in range(len(overview.kinds))]

    # Who each visit is compared with, modality by modality: the visits with
    # recordings of that modality that share its session label, when there
    # are enough of them, else every visit with that modality. An MRI kind
    # is never compared with a subject who only had an EEG, and a visit
    # with nothing to convert is compared with no one.
    nv, nk = len(overview.visits), len(overview.kinds)
    kind_modality = []
    for k in range(nk):
        seen = Counter(modality_of[p] for p in overview.kind_rows[k])
        kind_modality.append(seen.most_common(1)[0][0] if seen else "")
    has: dict[int, set] = defaultdict(set)
    for (i, k) in taken:
        has[i].add(kind_modality[k])
    groups: dict[tuple[int, str], tuple[int, ...]] = {}
    for modality in set(kind_modality):
        holders = [i for i in range(nv) if modality in has[i]]
        by_session: dict[str, list[int]] = defaultdict(list)
        for i in holders:
            by_session[overview.visits[i].session].append(i)
        for i in holders:
            same = by_session[overview.visits[i].session]
            group = same if len(same) >= MIN_PEERS else holders
            if len(group) >= MIN_PEERS:
                groups[(i, modality)] = tuple(group)
    overview.compared = bool(groups)

    cells: dict[tuple[int, int], Cell] = {}
    for k in range(nk):
        modality = kind_modality[k]
        usual_in: dict[tuple[int, ...], Optional[int]] = {}
        for group in {g for (_i, m), g in groups.items() if m == modality}:
            counts = Counter(len(taken.get((i, k), ())) for i in group)
            count, freq = counts.most_common(1)[0]
            usual_in[group] = count if freq >= USUAL_SHARE * len(group) else None
        for i in range(nv):
            have, out = tuple(taken.get((i, k), ())), tuple(dropped.get((i, k), ()))
            group = groups.get((i, modality))
            usual = usual_in.get(group) if group else None
            state = _state(len(have), usual, group, k, taken)
            if have or out or usual is not None:
                cells[(i, k)] = Cell(rows=have, left_out=out, usual=usual,
                                     peers=len(group) if group else 0, state=state)
    overview._cells = cells
    overview.findings = _findings(overview, rows, converting, groups, kind_modality)
    return overview


def _state(count: int, usual: Optional[int], group, k: int, taken) -> str:
    if usual is None or count == usual:
        return ""
    if usual >= 1:
        if count == 0:
            return "missing"
        return "fewer" if count < usual else "extra"
    holders = sum(1 for i in group if taken.get((i, k)))
    rare = len(group) >= RARE_MIN_PEERS and holders <= RARE_SHARE * len(group)
    return "rare" if rare else ""


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def _findings(ov: Overview, rows: list[_Row], converting: list[_Row],
              groups: dict[tuple[int, str], tuple[int, ...]],
              kind_modality: list[str]) -> list[Finding]:
    found: list[Finding] = []
    found += _table_findings(rows, converting)
    found += _visit_findings(ov, rows)
    found += _modality_findings(ov, rows)
    found += _id_like_findings(ov, converting)
    found += _cell_findings(ov, groups, kind_modality)
    order = {id(f): n for n, f in enumerate(found)}
    return sorted(found, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), order[id(f)]))


def _table_findings(rows: list[_Row], converting: list[_Row]) -> list[Finding]:
    out = []
    claimed: dict[tuple, list[int]] = defaultdict(list)
    for r in converting:
        claimed[(r.dataset, r.datatype, r.name)].append(r.pos)
    clashes = {key: pos for key, pos in claimed.items() if len(pos) > 1}
    if clashes:
        every = tuple(p for pos in clashes.values() for p in pos)
        if len(clashes) == 1:
            (key, pos), = clashes.items()
            text = (f"{len(pos)} recordings would be written as {key[2]}: one would "
                    "overwrite the other.")
        else:
            text = (f"{len(clashes)} names are each claimed by more than one recording "
                    f"({_listed([k[2] for k in clashes])}).")
        out.append(Finding("same-name", "err", text, every))
    unnamed = tuple(r.pos for r in rows if r.included and not (r.name and r.datatype))
    if unnamed:
        out.append(Finding(
            "unnamed", "err",
            f"{_amount(len(unnamed)).capitalize()} recording{'s' if len(unnamed) != 1 else ''} "
            "set to convert " + ("has" if len(unnamed) == 1 else "have")
            + " no BIDS name or datatype yet.", unnamed))
    check = tuple(r.pos for r in converting if r.check)
    if check:
        out.append(Finding(
            "to-check", "warn",
            f"{_amount(len(check)).capitalize()} name{'s' if len(check) != 1 else ''} "
            + ("is" if len(check) == 1 else "are")
            + " marked to check: the acquisition parameters did not settle "
            + ("it." if len(check) == 1 else "them."), check))
    return out


def _visit_findings(ov: Overview, rows: list[_Row]) -> list[Finding]:
    out = []
    converting_visits = {Visit(r.subject, r.session) for r in rows if r.included and r.name}

    # A visit whose every recording is left out.
    idle = [(i, v) for i, v in enumerate(ov.visits) if v not in converting_visits]
    if idle:
        visits = [v for _i, v in idle]
        rows_of = tuple(p for i, _v in idle for p in ov.visit_rows[i])
        if len(idle) == 1:
            text = (f"{visits[0].label} has nothing to convert: its "
                    f"{_amount(len(rows_of))} recording{'s are' if len(rows_of) != 1 else ' is'}"
                    " left out.")
        else:
            text = (f"{len(idle)} {_noun(visits, 2)} have nothing to convert "
                    f"({_listed([v.label for v in visits])}): all their recordings are left out.")
        out.append(Finding("nothing-to-convert", "info", text, rows_of))

    # Subjects without a session among subjects with sessions.
    active = [v for v in ov.visits if v in converting_visits]
    with_sessions = sorted({v.subject for v in active if v.session}, key=_natural)
    without = {v.subject for v in active if not v.session}
    only_without = sorted(without - set(with_sessions), key=_natural)
    mixed = sorted(without & set(with_sessions), key=_natural)
    labels = sorted({v.session for v in active if v.session}, key=_natural)
    shown = f" ({_listed(labels)})" if len(labels) <= NAMES_LISTED + 1 else ""

    def sessionless_rows(subjects) -> tuple[int, ...]:
        return tuple(p for i, v in enumerate(ov.visits)
                     if v.subject in subjects and not v.session for p in ov.visit_rows[i])

    if only_without and with_sessions:
        lead = (f"{only_without[0]} has no session" if len(only_without) == 1 else
                f"{len(only_without)} subjects have no session ({_listed(only_without)})")
        n = len(with_sessions)
        out.append(Finding(
            "no-session", "warn",
            f"{lead}, while the other {_amount(n)} {'has' if n == 1 else 'have'} sessions"
            f"{shown}.", sessionless_rows(only_without)))
    if mixed:
        lead = (f"{mixed[0]} has" if len(mixed) == 1 else
                f"{len(mixed)} subjects ({_listed(mixed)}) have")
        out.append(Finding(
            "with-and-without-session", "warn",
            f"{lead} recordings both with and without a session.", sessionless_rows(mixed)))

    # A session most subjects have and some do not.
    subjects_with = {s: {v.session for v in active if v.subject == s and v.session}
                     for s in with_sessions}
    if len(subjects_with) >= MIN_PEERS:
        holders = Counter(label for labels in subjects_with.values() for label in labels)
        for label, count in sorted(holders.items(), key=lambda kv: _natural(kv[0])):
            lacking = [s for s, labels in subjects_with.items() if label not in labels]
            if not lacking or count < MIN_PEERS or count < USUAL_SHARE * len(subjects_with):
                continue
            rows_of = tuple(p for i, v in enumerate(ov.visits)
                            if v.subject in lacking for p in ov.visit_rows[i])
            lead = (f"{lacking[0]} has" if len(lacking) == 1
                    else f"{len(lacking)} subjects ({_listed(lacking)}) have")
            out.append(Finding(
                f"missing-session:{label}", "warn",
                f"{lead} no {label}; the other {_amount(count)} "
                f"{'has' if count == 1 else 'have'} one.", rows_of))

    # Several sessions of one subject on one day.
    days: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r.included and r.session and r.day:
            days[r.subject][r.day].add(r.session)
    split = []
    for subject in sorted(days, key=_natural):
        for day, labels in sorted(days[subject].items()):
            if len(labels) >= 2:
                split.append((subject, day, sorted(labels, key=_natural)))
    if split:
        def rows_for(pairs) -> tuple[int, ...]:
            wanted = {(s, label) for s, _d, labels in pairs for label in labels}
            return tuple(p for i, v in enumerate(ov.visits)
                         if (v.subject, v.session) in wanted for p in ov.visit_rows[i])

        advice = " If they are one visit, give them one session label."
        if len(split) > SAME_DAY_LISTED:
            subjects = sorted({s for s, _d, _l in split}, key=_natural)
            out.append(Finding(
                "same-day", "info",
                f"{len(subjects)} subjects each have several sessions recorded on one day "
                f"({_listed(subjects)})." + advice, rows_for(split)))
        else:
            for subject, day, labels in split:
                out.append(Finding(
                    f"same-day:{subject}:{day}", "info",
                    f"{subject} has {len(labels)} sessions recorded on {day} "
                    f"({_listed(labels)})." + advice, rows_for([(subject, day, labels)])))
    return out


def _modality_findings(ov: Overview, rows: list[_Row]) -> list[Finding]:
    """A visit with no recording at all of a modality the others have: one
    finding, not one per kind it lacks."""
    present: dict[Visit, set] = defaultdict(set)
    for r in rows:
        if r.included and r.name and r.datatype:
            present[Visit(r.subject, r.session)].add(r.modality or "other")
    active = [i for i, v in enumerate(ov.visits) if present.get(v)]
    by_session: dict[str, list[int]] = defaultdict(list)
    for i in active:
        by_session[ov.visits[i].session].append(i)
    lacking: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i in active:
        same = by_session[ov.visits[i].session]
        group = same if len(same) >= MIN_PEERS else active
        if len(group) < MIN_PEERS:
            continue
        held = Counter(m for j in group for m in present[ov.visits[j]])
        for modality, count in held.items():
            if count >= USUAL_SHARE * len(group) and modality not in present[ov.visits[i]]:
                lacking[(modality, count)].append(i)
    out = []
    for (modality, count), visits in sorted(lacking.items()):
        name = _MODALITY_NAMES.get(modality, modality.upper())
        names = [ov.visits[i].label for i in visits]
        lead = (f"{names[0]} has" if len(names) == 1 else
                f"{len(names)} {_noun([ov.visits[i] for i in visits], 2)} ({_listed(names)}) have")
        others = count
        out.append(Finding(
            f"no-modality:{modality}", "warn",
            f"{lead} no {name} recordings; the other {_amount(others)} "
            f"{'has' if others == 1 else 'have'} some.",
            tuple(p for i in visits for p in ov.visit_rows[i])))
    return out


def _id_like_findings(ov: Overview, converting: list[_Row]) -> list[Finding]:
    out = []
    for key, distinct in sorted(ov.id_like.items()):
        carrying = tuple(r.pos for r in converting if any(k == key for k, _v in r.pairs))
        meaning = _ENTITY_MEANING.get(key, f"{key} labels")
        grouped = (" The grid groups them by datatype and suffix."
                   if ov.level == "suffix" or (ov.level == "task" and key != "task") else "")
        out.append(Finding(
            f"id-like:{key}", "warn",
            f"{distinct:,} different {key} labels across {len(carrying):,} recordings: they "
            f"look like file names or IDs rather than {meaning}." + grouped, carrying))
    return out


def _similar(a: str, b: str) -> bool:
    fa, fb = re.sub(r"[^0-9a-z]", "", a.lower()), re.sub(r"[^0-9a-z]", "", b.lower())
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    shorter = min(len(fa), len(fb))
    if shorter >= 3 and (fa.startswith(fb) or fb.startswith(fa)):
        return True
    return difflib.SequenceMatcher(None, fa, fb).ratio() >= SIMILAR_RATIO


def _spelled_apart(a: Kind, b: Kind) -> Optional[str]:
    """The entity ``a`` and ``b`` spell two ways, or ``None``."""
    if (a.datatype, a.suffix) != (b.datatype, b.suffix) or len(a.labels) != len(b.labels):
        return None
    differing = []
    for la, lb in zip(a.labels, b.labels):
        ka, _s, va = la.partition("-")
        kb, _s, vb = lb.partition("-")
        if ka != kb:
            return None
        if va != vb:
            differing.append((ka, va, vb))
    if len(differing) != 1:
        return None
    key, va, vb = differing[0]
    return key if _similar(va, vb) else None


def _cell_findings(ov: Overview, groups: dict[tuple[int, str], tuple[int, ...]],
                   kind_modality: list[str]) -> list[Finding]:
    out: list[Finding] = []
    nk, nv = len(ov.kinds), len(ov.visits)
    used: set[tuple[int, int]] = set()

    def matching(k: int, visits: Sequence[int]) -> int:
        """How many of the visits compared with ``visits`` have the usual."""
        others: set[int] = set()
        for i in visits:
            others.update(groups.get((i, kind_modality[k]), ()))
        others -= set(visits)
        return sum(1 for i in others
                   if ov.cell(i, k).usual is not None and ov.cell(i, k).count == ov.cell(i, k).usual)

    # One label spelled two ways: a visit lacks a kind the others have and
    # has one most of them do not, and the two differ in one similar label.
    pairs: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for i in range(nv):
        missing = [k for k in range(nk)
                   if ov.cell(i, k).state == "missing" and not ov.cell(i, k).left_out]
        odd = [k for k in range(nk) if ov.cell(i, k).count and ov.cell(i, k).usual in (0, None)]
        for m in missing:
            for u in odd:
                entity = None if (i, u) in used else _spelled_apart(ov.kinds[u], ov.kinds[m])
                if entity:
                    pairs[(u, m, entity)].append(i)
                    used.update({(i, u), (i, m)})
                    break
    for (u, m, entity), visits in pairs.items():
        names = [ov.visits[i].label for i in visits]
        verb = "has" if len(visits) == 1 else "have"
        text = (f"{_listed(names)} {verb} {ov.kinds[u].key} where the other "
                f"{_amount(matching(m, visits))} have {ov.kinds[m].key}: the same "
                f"{entity} spelled two ways?")
        rows = tuple(p for i in visits for p in ov.cell(i, u).rows)
        out.append(Finding(f"spelling:{ov.kinds[u].key}:{ov.kinds[m].key}", "warn", text, rows))

    for k, kind in enumerate(ov.kinds):
        flagged: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i in range(nv):
            cell = ov.cell(i, k)
            if cell.state and (i, k) not in used:
                state = cell.state
                if state == "missing" and cell.left_out:
                    state = "left-out"
                flagged[(state, cell.count)].append(i)
        for (state, count), visits in flagged.items():
            names = [ov.visits[i].label for i in visits]
            one = len(visits) == 1
            usual = ov.cell(visits[0], k).usual or 0
            others = _amount(matching(k, visits))
            if state == "left-out":
                rows = tuple(p for i in visits for p in ov.cell(i, k).left_out)
                lead = (f"{names[0]}'s {kind.key} is left out" if one else
                        f"{_listed(names)} leave {kind.key} out")
                text = f"{lead}; the other {others} convert theirs."
            elif state == "missing":
                rows = tuple(p for i in visits for p in ov.visit_rows[i])
                lead = (f"{names[0]} has no {kind.key}" if one else
                        f"{len(visits)} {_noun([ov.visits[i] for i in visits], 2)} have no "
                        f"{kind.key} ({_listed(names)})")
                text = f"{lead}; the other {others} have {_amount(usual)}."
            elif state in ("fewer", "extra"):
                rows = tuple(p for i in visits for p in ov.cell(i, k).rows)
                lead = (f"{names[0]} has {count} {kind.key}" if one else
                        f"{_listed(names)} have {count} {kind.key} each")
                text = f"{lead}; the other {others} have {_amount(usual)}."
            else:   # rare
                rows = tuple(p for i in visits for p in ov.cell(i, k).rows)
                verb = "has" if one else "have"
                text = f"Only {_listed(names)} {verb} {kind.key}."
            severity = "info" if state == "rare" else "warn"
            out.append(Finding(f"{state}:{kind.key}:{count}", severity, text, rows))
    return out


__all__ = [
    "Cell",
    "Finding",
    "Kind",
    "LEVELS",
    "LEVEL_TITLES",
    "Overview",
    "RecordingGroup",
    "Visit",
    "build_overview",
    "recording_groups",
]

"""What is actually in this dataset, in numbers.

The Editor can answer "is this file valid" for whichever file is selected, and
the validation pane can answer "what is wrong here". Neither answers the
question a user has when they open a dataset they did not make, or come back
to one after a month: what is in it, is it evenly filled, and where is the
work.

So this counts. Subjects, sessions, modalities, how much of the metadata each
datatype actually carries, and where the findings are concentrated. All of it
read from the tree and from a validation report the caller already has:
nothing here runs the validator, because drawing a summary is not a reason to
revalidate a dataset.

Two rules keep the numbers honest:

**Every figure is one a user could verify by hand.** No scores, no indexes, no
weighted composites. "41 of 60 declared fields are answered" can be checked;
"metadata completeness 68%" invites belief.

**An absence is reported as an absence.** A dataset with no participants table
does not get a zero, it gets a statement that there is no table, because those
are different facts and a zero reads as "measured and found empty".

Qt-free, so the figures can be tested without a GUI and reused by a report.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Files at the top of a dataset that describe it rather than belong to a
# subject. Counted separately, because folding them into "other" hides whether
# a dataset has the things that make it shareable.
DATASET_FILES = (
    "dataset_description.json", "participants.tsv", "participants.json",
    "README", "README.md", "README.txt", "CHANGES", "LICENSE",
    "CITATION.cff", ".bidsignore",
)


@dataclass
class ModalityRow:
    """One datatype, and how much of it there is."""

    datatype: str
    subjects: int = 0
    files: int = 0
    recordings: int = 0        # data files, not sidecars or tables
    errors: int = 0
    warnings: int = 0
    # Metadata: how many declared fields the sidecars of this datatype carry,
    # out of how many the standard declares for them.
    answered: int = 0
    declared: int = 0
    placeholders: int = 0      # fields whose value is still a marker

    @property
    def coverage(self) -> Optional[float]:
        """Answered over declared, or ``None`` when nothing is declared.

        ``None`` rather than 0.0: a datatype the schema declares nothing for
        has not scored badly, it has not been asked.
        """
        if not self.declared:
            return None
        return self.answered / self.declared


@dataclass
class SubjectRow:
    """One subject, and how evenly filled it is compared with the others."""

    label: str
    sessions: int = 0
    files: int = 0
    datatypes: tuple[str, ...] = ()
    errors: int = 0
    warnings: int = 0


@dataclass
class Dashboard:
    """Everything the dashboard shows, as data."""

    root: Path
    name: str = ""
    bids_version: str = ""
    subjects: list[SubjectRow] = dc_field(default_factory=list)
    modalities: list[ModalityRow] = dc_field(default_factory=list)
    # Dataset-level files that are present, and the ones that are not.
    present: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    participants: Optional[int] = None   # rows, or None when there is no table
    total_files: int = 0
    total_bytes: int = 0
    errors: int = 0
    warnings: int = 0
    validated: bool = False
    # The findings that occur most, so a user can see that four hundred
    # warnings are really one mistake made four hundred times.
    top_rules: list[tuple[str, int]] = dc_field(default_factory=list)

    @property
    def sessions(self) -> int:
        return sum(s.sessions for s in self.subjects)

    @property
    def outlier_subjects(self) -> list[tuple[str, tuple[str, ...]]]:
        """Subjects carrying something no majority of the others has.

        The mirror of :pyattr:`uneven_subjects`, and worth its own answer: one
        subject with an extra modality is usually a conversion that went
        further than the rest, or a file that landed in the wrong subject.
        """
        if len(self.subjects) < 2:
            return []
        counts: Counter = Counter()
        for subject in self.subjects:
            counts.update(set(subject.datatypes))
        rare = {
            datatype for datatype, n in counts.items()
            if n <= len(self.subjects) / 2
        }
        out = []
        for subject in self.subjects:
            extra = tuple(sorted(rare & set(subject.datatypes)))
            if extra:
                out.append((subject.label, extra))
        return out

    @property
    def uneven_subjects(self) -> list[str]:
        """Subjects that do not carry what the others do.

        The single most useful thing a dashboard can point at, because a
        subject missing a modality is invisible in a file tree and obvious in
        a table. Compared against the datatypes a MAJORITY of subjects have,
        so one over-complete subject does not make everyone else look wrong.
        """
        if len(self.subjects) < 2:
            return []
        counts: Counter = Counter()
        for subject in self.subjects:
            counts.update(set(subject.datatypes))
        majority = {
            datatype for datatype, n in counts.items()
            if n > len(self.subjects) / 2
        }
        return [
            s.label for s in self.subjects
            if majority - set(s.datatypes)
        ]


def build(root: Path, report=None) -> Dashboard:
    """Read the dataset, and the report if one is to hand."""
    root = Path(root)
    board = Dashboard(root=root)
    _read_description(root, board)
    counts = _finding_counts(report)
    board.validated = report is not None
    _walk_tree(root, board, counts)
    _read_participants(root, board)
    _read_dataset_files(root, board)
    _summarise_findings(report, board)
    return board


# ---------------------------------------------------------------------------


def _read_description(root: Path, board: Dashboard) -> None:
    try:
        data = json.loads((root / "dataset_description.json").read_text())
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        board.name = str(data.get("Name", "") or "")
        board.bids_version = str(data.get("BIDSVersion", "") or "")


def _finding_counts(report) -> dict[str, tuple[int, int]]:
    """``(errors, warnings)`` per relative path, mirrors excluded."""
    out: dict[str, tuple[int, int]] = {}
    for verdict in getattr(report, "files", []) or []:
        errors = warnings = 0
        for issue in verdict.issues or []:
            if getattr(issue, "mirrored", False):
                continue
            name = getattr(issue.severity, "value", str(issue.severity))
            if name == "err":
                errors += 1
            elif name == "warn":
                warnings += 1
        if errors or warnings:
            out[Path(verdict.path).as_posix()] = (errors, warnings)
    return out


def _walk_tree(root: Path, board: Dashboard, counts: dict) -> None:
    """One pass over the tree, filling the subject and modality tables."""
    from ..schema import list_datatypes

    known = set(list_datatypes())
    modalities: dict[str, ModalityRow] = {}
    seen_subjects: dict[str, set[str]] = {}

    for subject_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        row = SubjectRow(label=subject_dir.name)
        datatypes: set[str] = set()
        sessions = {
            p.name for p in subject_dir.glob("ses-*") if p.is_dir()
        }
        row.sessions = len(sessions)

        for path in subject_dir.rglob("*"):
            if not path.is_file() or ".bidsmgr" in path.parts:
                continue
            row.files += 1
            board.total_files += 1
            try:
                board.total_bytes += path.stat().st_size
            except OSError:
                pass
            rel = path.relative_to(root).as_posix()
            errors, warnings = counts.get(rel, (0, 0))
            row.errors += errors
            row.warnings += warnings

            datatype = path.parent.name
            if datatype not in known:
                continue
            datatypes.add(datatype)
            modality = modalities.setdefault(
                datatype, ModalityRow(datatype=datatype),
            )
            modality.files += 1
            modality.errors += errors
            modality.warnings += warnings
            if path.suffix == ".json":
                _count_metadata(path, modality)
            elif path.suffix not in (".tsv", ".bval", ".bvec"):
                modality.recordings += 1
            seen_subjects.setdefault(datatype, set()).add(subject_dir.name)

        row.datatypes = tuple(sorted(datatypes))
        board.subjects.append(row)

    for datatype, subjects in seen_subjects.items():
        modalities[datatype].subjects = len(subjects)
    board.modalities = sorted(modalities.values(), key=lambda m: m.datatype)


def _count_metadata(path: Path, modality: ModalityRow) -> None:
    """How much of what the standard declares this sidecar actually answers."""
    from ..metadata.engine import _NA_VALUE, _TODO_VALUE
    from ..schema import sidecar_fields

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    suffix = path.stem.rsplit("_", 1)[-1] if "_" in path.stem else path.stem
    try:
        declared = sidecar_fields(modality.datatype, suffix)
    except Exception:  # noqa: BLE001 - a summary must still be drawn
        return
    names = {field.name for field in declared}
    if not names:
        return
    modality.declared += len(names)
    for name in names:
        if name not in data:
            continue
        value = data[name]
        if _is_placeholder(value, _TODO_VALUE, _NA_VALUE):
            modality.placeholders += 1
        else:
            modality.answered += 1


def _is_placeholder(value, todo: str, na: str) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in (todo.upper(), na.upper())
    if isinstance(value, list):
        return bool(value) and all(
            _is_placeholder(v, todo, na) for v in value
        )
    return False


def _read_participants(root: Path, board: Dashboard) -> None:
    """Row count, or ``None`` when there is no table.

    The distinction matters: no table and an empty table are different facts,
    and a zero would read as the second.
    """
    path = root / "participants.tsv"
    if not path.exists():
        return
    try:
        lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return
    board.participants = max(len(lines) - 1, 0)


def _read_dataset_files(root: Path, board: Dashboard) -> None:
    """Which of the files that describe a dataset are there.

    A README is counted once however it is spelled. Listing README, README.md
    and README.txt as three separate absences reads as three problems and is
    one.
    """
    readmes = ("README", "README.md", "README.txt")
    present, absent = [], []
    for name in DATASET_FILES:
        if name in readmes:
            continue
        (present if (root / name).exists() else absent).append(name)
    if any((root / name).exists() for name in readmes):
        present.append("README")
    else:
        absent.append("README")
    board.present = tuple(sorted(present))
    board.absent = tuple(sorted(absent))


def _summarise_findings(report, board: Dashboard) -> None:
    if report is None:
        return
    rules: Counter = Counter()
    for verdict in getattr(report, "files", []) or []:
        for issue in verdict.issues or []:
            if getattr(issue, "mirrored", False):
                continue
            name = getattr(issue.severity, "value", str(issue.severity))
            if name == "err":
                board.errors += 1
            elif name == "warn":
                board.warnings += 1
            else:
                continue
            rules[issue.rule_id or "(unnamed)"] += 1
    for issue in getattr(report, "dataset_issues", []) or []:
        name = getattr(issue.severity, "value", str(issue.severity))
        if name == "err":
            board.errors += 1
        elif name == "warn":
            board.warnings += 1
        else:
            continue
        rules[issue.rule_id or "(unnamed)"] += 1
    board.top_rules = rules.most_common(8)


def human_bytes(size: int) -> str:
    """A size a person reads, not a number they convert."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


__all__ = [
    "DATASET_FILES",
    "Dashboard",
    "ModalityRow",
    "SubjectRow",
    "build",
    "human_bytes",
]

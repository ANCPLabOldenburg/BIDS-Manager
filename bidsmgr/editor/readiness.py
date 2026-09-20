"""Whether a dataset is fit to hand to somebody else.

A validator answers "is this legal BIDS". That is not the same question as "can
I share this", and the gap is where real datasets sit: no authors, a README
that is still the generated stub, a licence nobody chose, TODO placeholders the
conversion wrote and nobody replaced. None of those is an error. All of them
mean the dataset is not ready.

Each check answers one thing, says why it matters in a sentence a person can
act on, and where possible names the Editor action that fixes it. Nothing here
is a rule from the standard, so nothing here is presented as one: these are
BIDS Manager's opinions about sharing, and they are labelled as such.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# A README shorter than this is a stub, whatever it says. The scaffold writes
# one line; a dataset anyone can use needs a paragraph.
_README_MIN_CHARS = 160


@dataclass
class Check:
    """One thing that stands between a dataset and being shareable."""

    name: str
    passed: bool
    detail: str
    why: str
    fix: str = ""          # the Editor action that addresses it, if any

    @property
    def status(self) -> str:
        return "ready" if self.passed else "not ready"


def _stated(value) -> bool:
    """Is this an answer, or a note to self?

    ``License: "TODO"`` is what the metadata step writes when nothing could
    answer the field, and a checklist that reads it as a licence tells the
    user they are ready to share when they are not. One definition of "not an
    answer" is shared with the re-conversion merge, which faces the same
    question from the other side.
    """
    from ..metadata.preserve import is_placeholder

    return bool(value) and not is_placeholder(value)


def _load(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_dataset(root: Path, report=None) -> list[Check]:
    """Run every readiness check. ``report`` is used only if it is given."""
    root = Path(root)
    out: list[Check] = []
    description = _load(root / "dataset_description.json") or {}
    citation = (root / "CITATION.cff").exists()

    # Authorship. The single most common gap in datasets this tool produces.
    authors = description.get("Authors") or []
    if isinstance(authors, str):
        authors = [authors]
    authors = [a for a in authors if _stated(a)]
    has_authors = bool(authors) or citation
    out.append(Check(
        "Authorship", has_authors,
        f"{len(authors)} author(s) named" if authors else
        ("CITATION.cff present" if citation else "nobody is named"),
        "Nobody can credit a dataset with no authors, and journals ask for "
        "them at exactly the moment they are hardest to reconstruct.",
        "" if has_authors else "Fix ups: Write CITATION.cff",
    ))

    # Licence. Without one, a careful reader cannot legally use the data.
    licence = description.get("License") or ""
    out.append(Check(
        "Licence", _stated(licence),
        str(licence) if _stated(licence) else
        ("still says TODO" if licence else "not stated"),
        "Without a licence a reader has no permission to use the data, "
        "whatever your intent was.",
        "" if _stated(licence) else "Dataset metadata: License",
    ))

    # README, and whether it is still the stub.
    readme = next(
        (p for p in (root / "README", root / "README.md", root / "README.txt")
         if p.exists()), None,
    )
    if readme is None:
        out.append(Check(
            "README", False, "no README",
            "It is the first and often only thing a reader opens.",
            "",
        ))
    else:
        try:
            text = readme.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        long_enough = len(text) >= _README_MIN_CHARS
        out.append(Check(
            "README", long_enough,
            f"{len(text)} characters"
            + ("" if long_enough else ", which is still a stub"),
            "A generated one-liner tells a reader nothing about what was "
            "measured or why.",
            "",
        ))

    # CHANGES, which is how a reader knows which version they have.
    changes = root / "CHANGES"
    has_changes = changes.exists() and changes.stat().st_size > 0
    out.append(Check(
        "CHANGES", has_changes,
        "present" if has_changes else "absent or empty",
        "Without it nobody can tell which version of the dataset they have, "
        "including you in a year.",
        "",
    ))

    # participants.tsv, for anything with more than one subject.
    subjects = [p for p in root.glob("sub-*") if p.is_dir()]
    participants = root / "participants.tsv"
    ok = len(subjects) <= 1 or participants.exists()
    out.append(Check(
        "Participants table", ok,
        f"{len(subjects)} subject(s), "
        + ("table present" if participants.exists() else "no table"),
        "A dataset with several subjects and no participants table gives an "
        "analyst nothing to group or model by.",
        "" if ok else "Fix ups: Stamp TODO placeholders runs the metadata step",
    ))

    # TODO placeholders, which are notes to self and not values.
    todos = _count_todos(root)
    out.append(Check(
        "Placeholders", todos == 0,
        "none left" if not todos else f"{todos} field(s) still say TODO",
        "TODO is what BIDS Manager writes when nothing could answer a field. "
        "Shipping it tells a reader the metadata was never finished.",
        "" if not todos else "Validation: Whole dataset, then fix by group",
    ))

    # Validator errors, when a report is to hand. Not re-run: this is a
    # summary, and running the validator to draw a checklist would be absurd.
    if report is not None:
        errors = _count_errors(report)
        out.append(Check(
            "Validation", errors == 0,
            "no errors" if not errors else f"{errors} error(s)",
            "An error means a tool reading the dataset will get it wrong or "
            "refuse it.",
            "" if not errors else "Validation: Whole dataset",
        ))
    return out


def _count_todos(root: Path, limit: int = 5000) -> int:
    n = 0
    for p in root.rglob("*.json"):
        if ".bidsmgr" in p.parts:
            continue
        data = _load(p)
        if not data:
            continue
        for value in data.values():
            if isinstance(value, str) and value.strip().upper() == "TODO":
                n += 1
                if n >= limit:
                    return n
    return n


def _count_errors(report) -> int:
    from .types import Severity

    n = sum(
        1 for issue in getattr(report, "dataset_issues", []) or []
        if issue.severity is Severity.ERR
    )
    for verdict in getattr(report, "files", []) or []:
        n += sum(
            1 for issue in verdict.issues or []
            if issue.severity is Severity.ERR
            and not getattr(issue, "mirrored", False)
        )
    return n


def summarise(checks: list[Check]) -> str:
    passed = sum(1 for c in checks if c.passed)
    if passed == len(checks):
        return "Ready to share."
    return f"{passed} of {len(checks)} checks pass."


__all__ = ["Check", "check_dataset", "summarise"]

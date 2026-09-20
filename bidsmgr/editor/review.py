"""Warnings somebody has looked at and decided to keep.

A dataset reaches a point where the remaining warnings are all deliberate: a
task with no events because it was resting state, a recommended field nobody
can answer because the scanner is gone. Today the only ways to stop being told
about them are to turn the whole severity off, which hides the next real one,
or to keep scrolling past them forever.

Accepting a finding records a decision: who looked at it, when, and why it is
fine. It is stored **inside the dataset**, under ``.bidsmgr/``, because the
decision is about the data and should travel with it. A reviewer who opens the
dataset next month sees that somebody already considered this.

Two rules keep it honest:

**Only warnings can be accepted.** An error means the dataset is wrong, and a
tool that lets you dismiss wrongness is a tool that produces broken datasets
quietly. The Editor offers acceptance on warnings only.

**An acceptance names the file and the finding**, not just the rule. Accepting
``SIDECAR_KEY_RECOMMENDED`` everywhere would silence a class of problem across
files nobody has looked at. Accepting it for one file is a judgement about
that file.

Qt-free.
"""

from __future__ import annotations

import getpass
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ACCEPTED_FILE = Path(".bidsmgr") / "accepted_findings.json"


@dataclass(frozen=True)
class Acceptance:
    """One finding somebody decided to keep."""

    file: str            # relative path the finding is attached to
    rule_id: str
    field: str
    note: str
    who: str
    at: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.file, self.rule_id, self.field)

    def as_dict(self) -> dict:
        return {
            "file": self.file, "rule_id": self.rule_id, "field": self.field,
            "note": self.note, "who": self.who, "at": self.at,
        }


def _path(root: Path) -> Path:
    return Path(root) / ACCEPTED_FILE


def load(root: Path) -> dict[tuple[str, str, str], Acceptance]:
    """Every accepted finding, keyed by what it applies to."""
    path = _path(root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("cannot read %s: %s", path, exc)
        return {}
    out: dict[tuple[str, str, str], Acceptance] = {}
    for item in raw.get("accepted", []) if isinstance(raw, dict) else []:
        try:
            entry = Acceptance(
                file=str(item["file"]), rule_id=str(item["rule_id"]),
                field=str(item.get("field", "")), note=str(item.get("note", "")),
                who=str(item.get("who", "")), at=str(item.get("at", "")),
            )
        except (KeyError, TypeError):
            continue
        out[entry.key] = entry
    return out


def accept(
    root: Path, *, file: str, rule_id: str, field: str = "", note: str = "",
) -> Acceptance:
    """Record that somebody looked at this finding and kept it."""

    root = Path(root)
    entry = Acceptance(
        file=file, rule_id=rule_id, field=field or "", note=note,
        who=_who(), at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    current = load(root)
    current[entry.key] = entry
    _write(root, current, f"Accept {rule_id} in {file}")
    return entry


def unaccept(root: Path, *, file: str, rule_id: str, field: str = "") -> bool:
    """Undo a decision. Returns ``True`` when there was one."""
    root = Path(root)
    current = load(root)
    key = (file, rule_id, field or "")
    if key not in current:
        return False
    del current[key]
    _write(root, current, f"Stop accepting {rule_id} in {file}")
    return True


def _write(root: Path, entries: dict, label: str) -> None:
    from ..project.operations import begin_operation

    payload = {
        "note": (
            "Findings a reviewer looked at and decided to keep. Only "
            "warnings can be accepted; an error means the dataset is wrong."
        ),
        "accepted": [e.as_dict() for e in entries.values()],
    }
    try:
        with begin_operation(root, label) as op:
            op.write_json(_path(root), payload)
    except Exception as exc:  # noqa: BLE001 - fall back to a plain write
        log.debug("operation log unavailable (%s); writing directly", exc)
        target = _path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8",
        )


def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user on some CI images
        return "unknown"


def is_accepted(
    accepted: dict, *, file: str, rule_id: str, field: Optional[str] = "",
) -> Optional[Acceptance]:
    """The decision covering this finding, if there is one."""
    return accepted.get((file, rule_id, field or ""))


__all__ = [
    "ACCEPTED_FILE",
    "Acceptance",
    "accept",
    "is_accepted",
    "load",
    "unaccept",
]

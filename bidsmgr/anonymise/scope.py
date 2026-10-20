"""Which part of a dataset an anonymisation covers.

The files at the top (``participants.tsv`` and its JSON, inherited sidecars)
and which subjects and sessions. A table that lists participants or sessions
is changed only in the rows of those inside the scope, so anonymising two
subjects leaves the others' rows as they were.

Anonymising part of a participant's sessions now and the rest later keeps
their intervals: the number of weeks a participant's dates move is chosen
from ALL of their dates the first time, and remembered.

Qt-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Scope:
    """``None`` means all: every subject, every session of a chosen subject."""

    #: The files at the top of the dataset (participants.tsv and the like).
    top: bool = True
    #: ``sub-`` folders, or None for every subject.
    subjects: Optional[frozenset[str]] = None
    #: ``sub-01/ses-1`` pairs. A subject with none listed keeps all its
    #: sessions; one with some listed keeps only those.
    sessions: Optional[frozenset[str]] = None

    @property
    def whole(self) -> bool:
        return self.top and self.subjects is None and self.sessions is None

    def has_subject(self, subject: str) -> bool:
        return self.subjects is None or subject in self.subjects

    def has_session(self, subject: str, session: str) -> bool:
        if not self.has_subject(subject):
            return False
        if self.sessions is None:
            return True
        chosen = {s for s in self.sessions if s.startswith(subject + "/")}
        return not chosen or f"{subject}/{session}" in chosen

    def has_path(self, root: Path, path: Path) -> bool:
        """Whether a file of the dataset is inside the scope."""
        try:
            parts = Path(path).relative_to(root).parts
        except ValueError:
            return False
        if not parts or not parts[0].startswith("sub-"):
            return self.top
        if len(parts) > 2 and parts[1].startswith("ses-"):
            return self.has_session(parts[0], parts[1])
        return self.has_subject(parts[0])

    def describe(self) -> str:
        """The scope in words, for a dialog, a log or the history."""
        if self.whole:
            return "the whole dataset"
        bits = []
        if self.subjects is not None:
            n = len(self.subjects)
            bits.append(f"{n} subject" + ("" if n == 1 else "s"))
        else:
            bits.append("every subject")
        if self.sessions is not None:
            n = len(self.sessions)
            bits[-1] += f" ({n} session" + ("" if n == 1 else "s") + ")"
        if self.top:
            bits.append("the dataset files")
        return " and ".join(bits)


def scope_of(root: Path, paths: list[Path]) -> Scope:
    """The subjects and sessions ``paths`` are in (what was right-clicked),
    with the dataset files (their rows of participants.tsv). A path above
    every subject is the whole dataset."""
    root = Path(root)
    subjects: set[str] = set()
    sessions: set[str] = set()
    whole = set()
    for path in paths:
        try:
            parts = Path(path).relative_to(root).parts
        except ValueError:
            continue
        if not parts or not parts[0].startswith("sub-"):
            return Scope()
        if len(parts) > 1 and parts[1].startswith("ses-"):
            sessions.add(f"{parts[0]}/{parts[1]}")
        else:
            whole.add(parts[0])
        subjects.add(parts[0])
    if not subjects:
        return Scope()
    # A subject picked whole keeps all its sessions, whatever else was picked.
    sessions = {s for s in sessions if s.split("/")[0] not in whole}
    return Scope(subjects=frozenset(subjects),
                 sessions=frozenset(sessions) if sessions else None)


@dataclass(frozen=True)
class Unit:
    """A subject, its sessions and how many files each holds."""

    subject: str
    files: int
    #: ``(session, files)`` in name order; empty when it has none.
    sessions: tuple[tuple[str, int], ...] = ()


def dataset_units(root: Path) -> tuple[int, list[Unit]]:
    """``(files at the top, subjects)`` of the dataset: what a scope picks
    from. The folders that are not the raw dataset (``sourcedata/``,
    ``derivatives/``, ``code/``) and the tool's own are not counted."""
    root = Path(root)
    skip = {"sourcedata", "derivatives", "code"}

    def count(folder: Path) -> int:
        n = 0
        for dirpath, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            n += sum(1 for f in files if not f.startswith("."))
        return n

    top = 0
    units: list[Unit] = []
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return 0, []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            top += 1
        elif entry.is_dir() and entry.name.startswith("sub-"):
            sessions = []
            loose = 0
            for child in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if child.name.startswith("."):
                    continue
                if child.is_dir() and child.name.startswith("ses-"):
                    sessions.append((child.name, count(Path(child.path))))
                elif child.is_dir():
                    loose += count(Path(child.path))
                else:
                    loose += 1
            units.append(Unit(entry.name, loose + sum(n for _s, n in sessions),
                              tuple(sessions)))
        elif entry.is_dir() and entry.name not in skip:
            top += count(Path(entry.path))
    return top, units


__all__ = ["Scope", "Unit", "dataset_units", "scope_of"]

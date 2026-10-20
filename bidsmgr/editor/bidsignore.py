"""What ``.bidsignore`` says, what it does to this dataset, and what to offer
for the files a user picked.

A pattern file is written blind: the user types ``derivatives/`` and finds out
whether it worked by running the validator and reading what disappeared. The
useful questions are **which files does this pattern hide right now**, and,
going the other way, **which pattern hides exactly what I picked**. A pattern
matching nothing is almost always a typo; a pattern matching more than
intended is how a real problem gets silenced.

ONE matcher: the validator's (``vendor/bidsval/files/bidsignore``), so the
list, the tree and validation say the same thing. This module used to carry
its own translator, which read ``sub-01/anat`` as matching at any depth where
the validator (and git) anchor a pattern with a slash in its middle, and
which turned any picked file into its top folder: ignoring one stray file in
``sub-01/anat/`` ignored the whole subject (2026-10-19).

Every change is written at once as one step of the Editor's history.

Qt-free.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional

from ..vendor.bidsval.files.bidsignore import (
    DEFAULT_IGNORES,
    BidsIgnore,
    compile_pattern,
    load_bidsignore,
)

BIDSIGNORE = ".bidsignore"

#: Top-level folders every validator ignores whatever the file says.
STANDARD_FOLDERS: tuple[str, ...] = tuple(
    p.rstrip("/") for p in DEFAULT_IGNORES if p.endswith("/"))

#: Never listed: the tool's own state and version control.
_MACHINERY = frozenset({".bidsmgr", ".git", ".tmp_bidsmgr"})

_SUBJECT_OR_SESSION = re.compile(r"(?<![A-Za-z0-9])(sub|ses)-[A-Za-z0-9]+")


@dataclass
class IgnorePattern:
    """One line of the file, and what it currently matches."""

    raw: str
    comment: bool = False
    negate: bool = False
    matches: list[str] = dc_field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.matches)

    @property
    def is_redundant(self) -> bool:
        """Targets a dot-path or a folder the standard ignores already.

        Told apart from a dead pattern because it is not a mistake: the
        scaffold writes ``.bidsmgr/`` and it does no harm, it just does no
        work either. Calling it dead would send the user hunting for a typo
        that is not there.
        """
        if self.comment or self.negate:
            return False
        body = self.raw.strip().lstrip("/")
        return body.startswith(".") or body.rstrip("/") in STANDARD_FOLDERS

    @property
    def is_dead(self) -> bool:
        """Matches nothing, and is not merely redundant. Usually a typo."""
        if self.comment or not self.raw.strip() or self.is_redundant:
            return False
        return not self.matches


@dataclass(frozen=True)
class Listing:
    """The files a pattern can concern, and the folders nothing needs to."""

    #: Every file, relative and POSIX, sorted; hidden files and the folders
    #: the standard ignores are left out.
    paths: list[str]
    #: The top-level folders here that every validator ignores anyway.
    standard: list[str]


def list_dataset(root: Path) -> Listing:
    """One walk of the dataset. ``sourcedata/`` can hold a hundred thousand
    DICOM files, so the folders the standard ignores are named, not walked."""
    root = Path(root)
    paths: list[str] = []
    standard: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir == "."
        keep = []
        for name in sorted(dirs):
            if name.startswith(".") or name in _MACHINERY:
                continue
            if top and name in STANDARD_FOLDERS:
                standard.append(name)
                continue
            keep.append(name)
        dirs[:] = keep
        prefix = "" if top else rel_dir.replace(os.sep, "/") + "/"
        for name in files:
            if not name.startswith("."):
                paths.append(prefix + name)
    paths.sort()
    return Listing(paths=paths, standard=standard)


def dataset_paths(root: Path) -> list[str]:
    """Every file a pattern can concern (see :func:`list_dataset`)."""
    return list_dataset(root).paths


def _lines(root: Path) -> list[str]:
    target = Path(root) / BIDSIGNORE
    try:
        return target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def read_patterns(root: Path, paths: Optional[list[str]] = None) -> list[IgnorePattern]:
    """Parse ``.bidsignore`` and count what each line matches on its own."""
    paths = dataset_paths(root) if paths is None else paths
    out: list[IgnorePattern] = []
    for line in _lines(root):
        compiled = compile_pattern(line)
        item = IgnorePattern(raw=line.rstrip("\n"), comment=compiled is None)
        if compiled is not None:
            item.negate, rx = compiled
            item.matches = [p for p in paths if rx.match(p)]
        out.append(item)
    return out


def ignored_by(root: Path, paths: Iterable[str]) -> dict[str, str]:
    """``{path: the line that hides it}`` for every path the file hides."""
    rules = BidsIgnore(_lines(root), include_defaults=False)
    out: dict[str, str] = {}
    for path in paths:
        line = rules.deciding(path)
        if line is not None:
            out[path] = line
    return out


def ignored_paths(root: Path) -> set[str]:
    """Every path the file currently hides."""
    return set(ignored_by(root, dataset_paths(root)))


def matcher(root: Path) -> BidsIgnore:
    """What validation ignores here, the standard's folders included."""
    return load_bidsignore(root)


def matches_for(pattern: str, paths: list[str]) -> list[str]:
    """What a single pattern would match, for previewing before adding it."""
    compiled = compile_pattern(pattern)
    if compiled is None:
        return []
    return [p for p in paths if compiled[1].match(p)]


def findings_hidden_by(report, ignored: Iterable[str]) -> int:
    """How many findings of ``report`` are on files the file now hides.

    Counted against a report the caller already has, because running the
    validator to answer a display question would be absurd. A report made
    before a pattern was added still carries the findings it will drop.
    """
    if report is None:
        return 0
    ignored = set(ignored)
    if not ignored:
        return 0
    n = 0
    for verdict in getattr(report, "files", []) or []:
        if Path(verdict.path).as_posix() in ignored:
            n += len(verdict.issues or [])
    return n


# ---------------------------------------------------------------------------
# What to offer for a selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IgnoreChoice:
    """One way of ignoring (or no longer ignoring) what was picked."""

    #: exact, recording, folder, everywhere, extension (lines to ADD); stop
    #: (lines to remove, exceptions to add); remove (one line to remove).
    key: str
    label: str
    patterns: tuple[str, ...]
    #: Files of the dataset the choice concerns: hidden by an ignoring
    #: choice, brought back by a stop, hidden by the line a remove takes out.
    files: int
    #: ``!`` lines a stop adds, for a file a wider line also hides.
    exceptions: tuple[str, ...] = ()

    @property
    def ignores(self) -> bool:
        return self.key not in ("stop", "remove")


def _stem(name: str) -> str:
    """A BIDS name has one dot, before its extension."""
    return name.split(".", 1)[0]


def _extension(name: str) -> str:
    parts = name.split(".", 1)
    return "." + parts[1] if len(parts) == 2 else ""


def _bids_extensions() -> set[str]:
    from ..schema.loader import get_schema
    from ..vendor.bidsval.schema.introspect import extensions

    try:
        return {e.lower() for e in extensions(get_schema())}
    except Exception:  # noqa: BLE001 - offering less is the safe failure
        return set()


def _count(patterns: Iterable[str], paths: list[str]) -> set[str]:
    hit: set[str] = set()
    for pattern in patterns:
        hit.update(matches_for(pattern, paths))
    return hit


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def choices_for(selected: list[str], paths: list[str]) -> list[IgnoreChoice]:
    """The ways to ignore ``selected`` (relative POSIX paths; a folder ends
    with ``/``), narrowest first, each with how many files it would hide.

    Exactly what was picked always comes first. The wider ones are offered
    only when they hide more than the narrower before them: a choice that
    hides the same files under another name is noise.
    """
    picked = [s for s in selected if s and s.strip("/")]
    if not picked:
        return []
    files = [s for s in picked if not s.endswith("/")]
    folders = [s for s in picked if s.endswith("/")]
    out: list[IgnoreChoice] = []
    seen: list[tuple[set[str], int]] = []

    def offer(key: str, label: str, patterns: list[str]) -> None:
        # Offered when it hides other files, or the same ones in fewer
        # lines (one sub-* line also covers the subjects still to come).
        patterns = list(dict.fromkeys(patterns))
        hit = _count(patterns, paths)
        if any(hit == files and len(patterns) >= lines for files, lines in seen):
            return
        seen.append((hit, len(patterns)))
        out.append(IgnoreChoice(key, label, tuple(patterns), len(hit)))

    # 1. Exactly what was picked.
    if len(picked) == 1:
        label = "This folder" if folders else "This file"
    elif not folders:
        label = f"These {len(picked)} files"
    else:
        label = f"These {len(picked)} items"
    offer("exact", label, ["/" + s for s in picked])

    if files:
        # 2. With everything that shares its name: the sidecar, the bvals.
        recordings = ["/" + (os.path.dirname(f) + "/" if "/" in f else "")
                      + _stem(os.path.basename(f)) + ".*" for f in files]
        sidecar = any(p.endswith(".json") and p not in files
                      for p in _count(recordings, paths))
        offer("recording", ("With its sidecar" if len(files) == 1 else "With their sidecars")
              if sidecar else "With the files of the same name", recordings)

        # 3. The folder they are in, when they share one.
        parents = {os.path.dirname(f) for f in files}
        if len(parents) == 1 and not folders and "" not in parents:
            parent = parents.pop()
            offer("folder", f"The whole folder {parent}/", ["/" + parent + "/"])

    # 4. The same name in every subject and session.
    general = [_SUBJECT_OR_SESSION.sub(lambda m: m.group(1) + "-*", s) for s in picked]
    if general != picked:
        where = ("every subject and session"
                 if any("ses-*" in g for g in general) else "every subject")
        offer("everywhere", f"The same name in {where}", ["/" + g for g in general])

    # 5. Every file of an extension the standard does not use.
    exts = {_extension(os.path.basename(f)).lower() for f in files}
    exts.discard("")
    if len(exts) == 1:
        ext = exts.pop()
        if ext not in _bids_extensions():
            offer("extension", f"Every {ext} file", ["*" + ext])
    return out


def stop_choices(selected: list[str], paths: list[str],
                 root: Path) -> list[IgnoreChoice]:
    """The ways to have ``selected`` validated again.

    First, when it can: ONE choice for everything picked, "Stop ignoring these
    N files". It removes the lines that hide nothing else, and adds an
    exception (``!/path``) for a file a wider line also hides, so the rest of
    what that line hides stays hidden. An exception is offered only where the
    validator honours it: a file inside an ignored folder cannot be brought
    back that way (gitignore's rule), and is left to the second kind.

    Then one choice per wider line: remove it, saying how much else it hides.
    """
    picked = [s for s in selected if s and s.strip("/")]
    if not picked:
        return []

    def under(path: str) -> bool:
        return any(path == s or (s.endswith("/") and path.startswith(s)) for s in picked)

    concerned = [p for p in paths if under(p)]
    hidden = sorted(ignored_by(root, concerned))
    if not hidden:
        return []
    lines = _lines(root)
    wanted, hidden_set = set(concerned), set(hidden)
    # Every line that hides a picked file: it goes when it hides nothing
    # else (own), it stays when it does (wide).
    own, wide = [], []
    for line in dict.fromkeys(line.strip() for line in lines):
        compiled = compile_pattern(line)
        if compiled is None or compiled[0]:
            continue
        hits = set(matches_for(line, paths))
        if hits & hidden_set:
            (own if hits <= wanted else wide).append(line)
    kept = [line for line in lines if line.strip() not in set(own)]
    still = BidsIgnore(kept, include_defaults=False)
    exceptions = [f"!/{p}" for p in hidden if still.deciding(p) is not None]
    after = BidsIgnore(kept + exceptions, include_defaults=False)
    back = [p for p in hidden if after.deciding(p) is None]
    # An exception the validator would not honour (a file in a folder a line
    # hides) is not written.
    exceptions = tuple(e for e in exceptions if e[2:] in set(back))
    out: list[IgnoreChoice] = []
    if back:
        n = len(hidden)
        label = ("Stop ignoring this file" if n == 1
                 else f"Stop ignoring these {n} files" if len(back) == n
                 else f"Stop ignoring {len(back)} of these {n} files")
        out.append(IgnoreChoice("stop", label, tuple(own), len(back), exceptions))
    for line in wide:
        total = len(matches_for(line, paths))
        out.append(IgnoreChoice("remove", f"Remove {line}, which hides "
                                          f"{_plural(total, 'file')}", (line,), total))
    return out


def apply_choice(root: Path, choice: IgnoreChoice) -> tuple[list[str], list[str]]:
    """Make ``choice`` as ONE step of the history: ``(lines added, lines
    removed)``, both empty when there was nothing to do."""
    lines = _lines(root)
    if choice.ignores:
        add, drop = list(choice.patterns), set()
    else:
        add, drop = list(choice.exceptions), {p.strip() for p in choice.patterns}
    kept = [line for line in lines if line.strip() not in drop]
    removed = [line.strip() for line in lines if line.strip() in drop]
    have = {line.strip() for line in kept}
    added = [a.strip() for a in dict.fromkeys(add) if a.strip() and a.strip() not in have]
    if not added and not removed:
        return [], []
    if choice.ignores:
        what = added[0] if len(added) == 1 else _plural(len(added), "pattern")
        label = f"Ignore in validation: {what}"
    elif choice.key == "stop":
        label = f"Stop ignoring {_plural(choice.files, 'file')}"
    else:
        label = f"Stop ignoring: {removed[0] if removed else added[0]}"
    write_patterns(root, kept + added, label=label)
    return added, removed


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_patterns(root: Path, lines: list[str], *, label: str = "Edit .bidsignore") -> Path:
    """Save the pattern list as one step of the Editor's history."""
    from ..project.operations import begin_operation

    root = Path(root)
    target = root / BIDSIGNORE
    text = "\n".join(line.rstrip() for line in lines).rstrip("\n") + "\n"
    with begin_operation(root, label) as op:
        op.write_text(target, text)
    return target


def add_patterns(root: Path, patterns: Iterable[str], *,
                 label: Optional[str] = None) -> list[str]:
    """Append ``patterns`` the file does not already have. Returns those
    added (none: nothing written)."""
    lines = _lines(root)
    have = {line.strip() for line in lines}
    new = [p.strip() for p in dict.fromkeys(patterns) if p.strip() and p.strip() not in have]
    if new:
        what = new[0] if len(new) == 1 else _plural(len(new), "pattern")
        write_patterns(root, lines + new, label=label or f"Ignore in validation: {what}")
    return new


def remove_patterns(root: Path, patterns: Iterable[str], *,
                    label: Optional[str] = None) -> list[str]:
    """Remove every line equal to one of ``patterns``. Returns those removed."""
    drop = {p.strip() for p in patterns if p.strip()}
    lines = _lines(root)
    kept = [line for line in lines if line.strip() not in drop]
    gone = [line.strip() for line in lines if line.strip() in drop]
    if gone:
        what = gone[0] if len(gone) == 1 else _plural(len(gone), "pattern")
        write_patterns(root, kept, label=label or f"Stop ignoring: {what}")
    return gone


__all__ = [
    "BIDSIGNORE",
    "apply_choice",
    "STANDARD_FOLDERS",
    "IgnoreChoice",
    "IgnorePattern",
    "Listing",
    "add_patterns",
    "choices_for",
    "dataset_paths",
    "findings_hidden_by",
    "ignored_by",
    "ignored_paths",
    "list_dataset",
    "matcher",
    "matches_for",
    "read_patterns",
    "remove_patterns",
    "stop_choices",
    "write_patterns",
]

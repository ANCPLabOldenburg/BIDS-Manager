"""What ``.bidsignore`` says, and what it actually does to this dataset.

A pattern file is written blind: the user types ``derivatives/`` and finds out
whether it worked by running the validator and reading what disappeared. The
useful question, which nothing answers today, is **which files does this
pattern match right now**. A pattern matching nothing is almost always a typo,
and a pattern matching more than intended is how a real problem gets silenced.

So the model here is not a text file, it is a list of patterns each carrying
its current match count, computed against the dataset on disk.

Matching follows the ``.gitignore``-style rules BIDS specifies: a trailing
slash matches a directory and everything under it, a leading slash anchors to
the dataset root, ``*`` does not cross a path separator and ``**`` does.

Qt-free.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BIDSIGNORE = ".bidsignore"


@dataclass
class IgnorePattern:
    """One line of the file, and what it currently matches."""

    raw: str
    comment: bool = False
    matches: list[str] = dc_field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.matches)

    @property
    def is_redundant(self) -> bool:
        """Targets a dot-path, which every BIDS tool ignores anyway.

        Told apart from a dead pattern because it is not a mistake: the
        scaffold writes ``.bidsmgr/`` and it does no harm, it just does no
        work either. Calling it dead would send the user hunting for a typo
        that is not there.
        """
        return not self.comment and self.raw.strip().lstrip("/").startswith(".")

    @property
    def is_dead(self) -> bool:
        """Matches nothing, and is not merely redundant. Usually a typo."""
        if self.comment or not self.raw.strip() or self.is_redundant:
            return False
        return not self.matches


def _to_regex(pattern: str) -> Optional[re.Pattern]:
    """Translate one gitignore-style pattern into a regex over relative paths."""
    pat = pattern.strip()
    if not pat or pat.startswith("#"):
        return None
    anchored = pat.startswith("/")
    pat = pat.lstrip("/")
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    if not pat:
        return None

    # fnmatch's ``*`` crosses separators, which gitignore's does not. Build the
    # expression by hand so ``*`` and ``**`` mean what the user expects.
    out: list[str] = []
    i = 0
    while i < len(pat):
        ch = pat[i]
        if pat.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            close = pat.find("]", i)
            if close == -1:
                out.append(re.escape(ch))
            else:
                out.append(fnmatch.translate(pat[i:close + 1])[4:-3])
                i = close + 1
                continue
        else:
            out.append(re.escape(ch))
        i += 1
    body = "".join(out)
    prefix = "" if anchored else "(?:.*/)?"
    tail = "(?:/.*)?" if dir_only else "(?:/.*)?"
    try:
        return re.compile(f"^{prefix}{body}{tail}$")
    except re.error as exc:
        log.debug("bad ignore pattern %r: %s", pattern, exc)
        return None


def dataset_paths(root: Path) -> list[str]:
    """Every file in the dataset as a relative posix path.

    ``.bidsmgr`` is left out: it is the tool's own state, every BIDS tool
    ignores dotfiles already, and listing it would invite someone to add a
    pattern that is not needed.
    """
    root = Path(root)
    out: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".bidsmgr" in p.parts or ".git" in p.parts:
            continue
        try:
            out.append(p.relative_to(root).as_posix())
        except ValueError:
            continue
    return sorted(out)


def read_patterns(root: Path, paths: Optional[list[str]] = None) -> list[IgnorePattern]:
    """Parse ``.bidsignore`` and count what each line matches."""
    root = Path(root)
    target = root / BIDSIGNORE
    if not target.exists():
        return []
    paths = dataset_paths(root) if paths is None else paths
    out: list[IgnorePattern] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        is_comment = stripped.startswith("#") or not stripped
        item = IgnorePattern(raw=line.rstrip("\n"), comment=is_comment)
        if not is_comment:
            rx = _to_regex(line)
            if rx is not None:
                item.matches = [p for p in paths if rx.match(p)]
        out.append(item)
    return out


def matches_for(pattern: str, paths: list[str]) -> list[str]:
    """What a single pattern would match, for previewing before adding it."""
    rx = _to_regex(pattern)
    return [p for p in paths if rx.match(p)] if rx is not None else []


def ignored_paths(root: Path) -> set[str]:
    """Every path currently ignored, for showing the effect as a whole."""
    paths = dataset_paths(root)
    out: set[str] = set()
    for item in read_patterns(root, paths):
        out.update(item.matches)
    return out


def findings_hidden_by(root: Path, report) -> int:
    """How many current findings the ignore file is suppressing.

    The point of a pattern is to silence something, so the pane should be able
    to say what. Counted against a report the caller already has, because
    running the validator to answer a display question would be absurd.
    """
    if report is None:
        return 0
    ignored = ignored_paths(root)
    if not ignored:
        return 0
    n = 0
    for verdict in getattr(report, "files", []) or []:
        rel = Path(verdict.path).as_posix()
        if rel in ignored:
            n += len(verdict.issues or [])
    return n


def suggest_pattern(rel_path: str) -> str:
    """A reasonable pattern for a path the user picked.

    A file inside a top-level folder that is not part of the BIDS tree is
    almost always wanted as a whole folder, not one file at a time.
    """
    parts = rel_path.split("/")
    if len(parts) > 1:
        return parts[0] + "/"
    return rel_path


def write_patterns(root: Path, lines: list[str]) -> Path:
    """Save the pattern list, reversibly."""
    from ..project.operations import begin_operation

    root = Path(root)
    target = root / BIDSIGNORE
    text = "\n".join(line.rstrip() for line in lines).rstrip("\n") + "\n"
    with begin_operation(root, "Edit .bidsignore") as op:
        op.write_text(target, text)
    return target


__all__ = [
    "BIDSIGNORE",
    "IgnorePattern",
    "dataset_paths",
    "findings_hidden_by",
    "ignored_paths",
    "matches_for",
    "read_patterns",
    "suggest_pattern",
    "write_patterns",
]

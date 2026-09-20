"""Read and apply a dataset's ``.bidsignore`` file.

``.bidsignore`` lets a dataset declare files that are intentionally outside BIDS, so
they are not reported as ``NOT_INCLUDED``. It uses gitignore-style patterns. This is
a pragmatic subset of gitignore semantics covering the patterns ``.bidsignore`` files
actually use: globs (``*``, ``**``, ``?``), directory patterns (a trailing ``/``),
root-anchored patterns (a leading ``/``), and negation (a leading ``!``).
"""

from __future__ import annotations

import re
from pathlib import Path

_SPECIAL = set(".^$+{}[]()|\\")

# Patterns the reference validator always ignores, prepended to any .bidsignore.
# These directories/files are never validated: version-control and hidden files,
# raw source data, analysis code, stimulus files, and logs.
DEFAULT_IGNORES = [".git**", ".*", "sourcedata/", "code/", "stimuli/", "log/"]


class BidsIgnore:
    """A compiled set of ignore patterns (the BIDS defaults plus any .bidsignore)."""

    def __init__(self, patterns: list[str], *, include_defaults: bool = True) -> None:
        self._rules: list[tuple[bool, re.Pattern[str]]] = []
        source = (DEFAULT_IGNORES + list(patterns)) if include_defaults else list(patterns)
        for raw in source:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:]
            self._rules.append((negate, _compile(line)))

    def match(self, relpath: str) -> bool:
        """Whether ``relpath`` (POSIX, dataset-relative) is ignored."""
        ignored = False
        for negate, pattern in self._rules:
            if pattern.match(relpath):
                ignored = not negate
        return ignored


def load_bidsignore(root: str | Path) -> BidsIgnore:
    """Load ``<root>/.bidsignore`` if present, always including the BIDS defaults."""
    path = Path(root) / ".bidsignore"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    return BidsIgnore(lines)


def _compile(pattern: str) -> re.Pattern[str]:
    anchored = pattern.startswith("/")
    body = pattern.strip("/")
    regex = []
    placeholder = body.replace("**", "\x00")  # protect the recursive wildcard
    for char in placeholder:
        if char == "\x00":
            regex.append(".*")
        elif char == "*":
            regex.append("[^/]*")
        elif char == "?":
            regex.append("[^/]")
        elif char in _SPECIAL:
            regex.append("\\" + char)
        else:
            regex.append(char)
    prefix = "^" if (anchored or "/" in body) else "(?:^|.*/)"
    # A pattern matches the named file/directory and anything beneath it.
    return re.compile(prefix + "".join(regex) + "(?:/.*)?$")

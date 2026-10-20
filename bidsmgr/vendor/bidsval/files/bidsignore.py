"""Read and apply a dataset's ``.bidsignore`` file.

``.bidsignore`` lets a dataset declare files that are intentionally outside BIDS, so
they are not reported as ``NOT_INCLUDED``. It uses gitignore-style patterns. This is
a pragmatic subset of gitignore semantics covering the patterns ``.bidsignore`` files
actually use: globs (``*``, ``**``, ``?``, ``[...]``), directory patterns (a trailing
``/``), root-anchored patterns (a leading ``/``, or a ``/`` in the middle), and
negation (a leading ``!``).

VENDORED CHANGE (BIDS Manager): ``compile_pattern`` is public and
``BidsIgnore.deciding`` names the line that decided, so the Editor's ignore list
and tree use THIS matcher rather than a second one that disagreed with it;
character classes ``[...]`` are read as gitignore reads them (they were matched
as literal brackets); and a file inside an ignored folder stays ignored
whatever a later ``!`` line says, as gitignore and the reference validator's
``ignore`` package rule ("It is not possible to re-include a file if a parent
directory of that file is excluded").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_SPECIAL = set(".^$+{}[]()|\\")

# Patterns the reference validator always ignores, prepended to any .bidsignore.
# These directories/files are never validated: version-control and hidden files,
# raw source data, analysis code, stimulus files, and logs.
DEFAULT_IGNORES = [".git**", ".*", "sourcedata/", "code/", "stimuli/", "log/"]


class BidsIgnore:
    """A compiled set of ignore patterns (the BIDS defaults plus any .bidsignore)."""

    def __init__(self, patterns: list[str], *, include_defaults: bool = True) -> None:
        self._rules: list[tuple[bool, re.Pattern[str], str]] = []
        #: A folder's answer, asked once: every file under it asks again.
        self._folders: dict[str, Optional[str]] = {}
        source = (DEFAULT_IGNORES + list(patterns)) if include_defaults else list(patterns)
        for raw in source:
            compiled = compile_pattern(raw)
            if compiled is not None:
                negate, regex = compiled
                self._rules.append((negate, regex, raw.rstrip()))

    def match(self, relpath: str) -> bool:
        """Whether ``relpath`` (POSIX, dataset-relative) is ignored."""
        return self.deciding(relpath) is not None

    def deciding(self, relpath: str) -> Optional[str]:
        """The pattern that leaves ``relpath`` ignored (the last matching line,
        as in gitignore, or the line that ignores a folder it is in), or None
        when it is not ignored."""
        parts = relpath.split("/")
        for depth in range(1, len(parts)):
            folder = "/".join(parts[:depth])
            if folder not in self._folders:
                self._folders[folder] = self._last(folder)
            if self._folders[folder] is not None:
                return self._folders[folder]
        return self._last(relpath)

    def _last(self, relpath: str) -> Optional[str]:
        decided: Optional[str] = None
        for negate, pattern, raw in self._rules:
            if pattern.match(relpath):
                decided = None if negate else raw
        return decided


def load_bidsignore(root: str | Path, *, include_defaults: bool = True) -> BidsIgnore:
    """Load ``<root>/.bidsignore`` if present, with the BIDS defaults unless told not."""
    path = Path(root) / ".bidsignore"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    return BidsIgnore(lines, include_defaults=include_defaults)


def compile_pattern(line: str) -> Optional[tuple[bool, re.Pattern[str]]]:
    """``(negated, regex)`` for one line of the file; None for a blank line or
    a comment."""
    line = line.rstrip()
    if not line or line.lstrip().startswith("#"):
        return None
    negate = line.startswith("!")
    if negate:
        line = line[1:]
    if not line.strip("/"):
        return None
    return negate, _compile(line)


def _compile(pattern: str) -> re.Pattern[str]:
    anchored = pattern.startswith("/")
    body = pattern.strip("/")
    regex = []
    placeholder = body.replace("**", "\x00")  # protect the recursive wildcard
    i = 0
    while i < len(placeholder):
        char = placeholder[i]
        if char == "\x00":
            regex.append(".*")
        elif char == "*":
            regex.append("[^/]*")
        elif char == "?":
            regex.append("[^/]")
        elif char == "[" and "]" in placeholder[i + 2:]:
            # A class, as gitignore reads it: [abc], [a-z], [!abc] or [^abc].
            close = placeholder.index("]", i + 2)
            inner = placeholder[i + 1:close]
            if inner[:1] in ("!", "^"):
                inner = "^" + inner[1:].replace("\\", "\\\\")
            else:
                inner = inner.replace("\\", "\\\\")
            regex.append(f"[{inner}]")
            i = close + 1
            continue
        elif char in _SPECIAL:
            regex.append("\\" + char)
        else:
            regex.append(char)
        i += 1
    prefix = "^" if (anchored or "/" in body) else "(?:^|.*/)"
    # A pattern matches the named file/directory and anything beneath it.
    return re.compile(prefix + "".join(regex) + "(?:/.*)?$")

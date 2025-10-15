"""Utility helpers for handling study names.

This module centralizes small helpers that operate on study names so the
behavior is consistent across the different entry points.  The helpers are
tiny, but placing them in their own module makes it easy to share the logic
between the GUI, command line utilities, and background data preparation
without creating import cycles.
"""

from __future__ import annotations

import re

# Regular expression that captures alphanumeric "words" inside a study name.
# The logic in :func:`normalize_study_name` walks through these matches and keeps
# track of consecutive duplicates.
_WORD_RE = re.compile(r"[0-9A-Za-z]+")


def normalize_study_name(raw: str) -> str:
    """Return ``raw`` with consecutive duplicate words removed.

    Study descriptions exported from PACS or spreadsheets occasionally contain
    repeated words (for example ``"study_study"`` or ``"BIDS BIDS"``).  These
    repetitions often come from manual copy/paste errors and later cause issues
    when the name is used as a folder on disk.  The function below removes such
    repeats **case-insensitively** while leaving the surrounding punctuation and
    spacing untouched so the resulting string still reads naturally for the
    user.

    Parameters
    ----------
    raw:
        Original study name as exported from metadata.  Non-string values are
        coerced to strings so the helper can safely operate on values from
        :mod:`pandas` data frames.

    Returns
    -------
    str
        The cleaned study name where direct repetitions of the same alphanumeric
        token have been collapsed into a single occurrence.  If *raw* contains
        no words the input is returned unchanged.
    """

    text = "" if raw is None else str(raw)
    text = text.strip()
    if not text:
        return ""

    pieces: list[str] = []
    last_word: str | None = None
    cursor = 0

    for match in _WORD_RE.finditer(text):
        start, end = match.span()

        # Preserve whatever characters appear between the previous word and the
        # current one (underscores, dashes, spaces, …) so the user sees the
        # exact same separators in the cleaned value.
        if start > cursor:
            separator = text[cursor:start]
            if separator:
                pieces.append(separator)

        word = match.group(0)
        lowered = word.lower()

        if lowered != last_word:
            # This word is different from the previous one (ignoring case), so
            # keep it and update our tracking state.
            pieces.append(word)
            last_word = lowered
        else:
            # When we detect a duplicate we also remove the separator that was
            # appended just above, preventing leftover characters such as
            # trailing underscores when turning ``"study_study"`` into
            # ``"study"``.
            if start > cursor and pieces:
                separator = text[cursor:start]
                if pieces[-1] == separator:
                    pieces.pop()

        cursor = end

    # Include any trailing punctuation that came after the final word.
    if cursor < len(text):
        pieces.append(text[cursor:])

    return "".join(pieces).strip()


__all__ = ["normalize_study_name"]


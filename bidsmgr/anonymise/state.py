"""What anonymising a dataset must remember between runs.

One number of days per participant, so a session converted next month is
shifted with its participant's earlier sessions and intervals survive; and a
secret, so a pseudonym (a UID, a device serial) is the same every time for the
same original and cannot be recomputed by anyone who has only the dataset.

Kept in ``.bidsmgr/project/anonymise.json``, beside the project's other state,
and written through the operation that applies the change, so an undo takes it
back with the files.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import secrets
import uuid
from dataclasses import dataclass, field
from pathlib import Path

#: Where the state lives, relative to the dataset root.
STATE_FILE = Path(".bidsmgr") / "project" / "anonymise.json"

#: Shifted dates land in these years (the standard: 1925 or earlier). The
#: first date of a participant lands in the first fourteen of them, which
#: leaves ten years for the rest of their sessions.
EARLIEST_YEAR = 1902
LATEST_YEAR = 1925


@dataclass
class State:
    secret: str = field(default_factory=lambda: secrets.token_hex(16))
    offsets: dict[str, int] = field(default_factory=dict)   # participant -> days back

    def to_json(self) -> str:
        return json.dumps({"version": 1, "secret": self.secret, "offsets": self.offsets},
                          indent=2, sort_keys=True) + "\n"

    # -- dates ----------------------------------------------------------

    def days_back(self, participant: str, earliest: _dt.date | None,
                  latest: _dt.date | None = None) -> int:
        """How many days ``participant``'s dates move back: remembered, or
        chosen now so their earliest date lands in 1902-1915 and their latest
        in 1925 or before, in whole weeks (a Monday stays a Monday)."""
        if participant in self.offsets:
            return self.offsets[participant]
        if earliest is None:
            return 0
        latest = latest or earliest
        digest = hashlib.sha256(f"{self.secret}|year|{participant}".encode()).digest()
        year = EARLIEST_YEAR + digest[0] % 14
        day = min(earliest.day, 28) if earliest.month == 2 else earliest.day
        days = (earliest - _dt.date(year, earliest.month, day)).days
        # A participant measured over more than a decade: move further, so
        # the latest date is still 1925 or before (the standard's rule wins
        # over landing the first date in a particular year).
        days = max(days, (latest - _dt.date(LATEST_YEAR, 12, 31)).days)
        days = -(-days // 7) * 7            # up to whole weeks
        self.offsets[participant] = max(0, days)
        return self.offsets[participant]

    # -- pseudonyms -------------------------------------------------------

    def _digest(self, kind: str, value: str) -> bytes:
        return hashlib.sha256(f"{self.secret}|{kind}|{value}".encode()).digest()

    def uid(self, value: str) -> str:
        """A DICOM UID for ``value``: ``2.25.`` and a 128-bit number (the
        form DICOM gives for UUID-derived UIDs), the same for the same UID."""
        return "2.25." + str(uuid.UUID(bytes=self._digest("uid", value)[:16]).int)

    def label(self, kind: str, value: str) -> str:
        """A short stable pseudonym: ``device-3f9a2c1b``."""
        return f"{kind}-{self._digest(kind, value).hex()[:8]}"


def load(root: Path) -> State:
    path = Path(root) / STATE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return State()
    if not isinstance(data, dict) or not data.get("secret"):
        return State()
    offsets = {str(k): int(v) for k, v in (data.get("offsets") or {}).items()
               if isinstance(v, int)}
    return State(secret=str(data["secret"]), offsets=offsets)


def is_shifted(date: _dt.date) -> bool:
    """Already anonymised: a date the standard's range puts in 1925 or before.
    No real recording is that old, and treating it so is what makes running
    the anonymisation twice change nothing more."""
    return date.year <= LATEST_YEAR


__all__ = ["EARLIEST_YEAR", "LATEST_YEAR", "STATE_FILE", "State", "is_shifted", "load"]

"""Time helpers shared across scan, fixups, and converter.

One function, and it has to cope with three spellings of the same fact
because three different producers write it:

===========================  ==============================  ==============
Form                         Example                         Written by
===========================  ==============================  ==============
DICOM TM                     ``093359.365000``               the scanner
DICOM DT                     ``20260917093359.365000``       Siemens XA
ISO, time only               ``09:33:59.365000``             dcm2niix
ISO, date and time           ``2026-09-17T09:33:59.365000``  dcm2niix
===========================  ==============================  ==============

The previous version parsed only the first and read the others by taking
their leading characters as hours, minutes and seconds. That is not a
failure to parse, it is a WRONG ANSWER that looks like a right one:
``2026-09-17T09:33:59`` came back as 20:26:00, and every caller trusted
it. The IntendedFor fixup then either discarded the time or ordered the
runs by a number that had nothing to do with when they were acquired.

So each form is now recognised explicitly and anything else returns
``None``. Guessing is what produced the defect.
"""

from __future__ import annotations

import re
from typing import Optional

# ``HH:MM:SS`` with optional fraction, anchored at the end so it also
# matches the tail of an ISO date-time.
#
# One or two digits per component, NOT two. dcm2niix writes the seconds
# unpadded when they are below ten: a real run in the test data carries
# ``10:17:1.572500``. Requiring two digits rejected it, the run dropped out
# of the timed set, and one untimed run makes the IntendedFor fixup fall
# back to pairing every fieldmap with every run.
_ISO_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{1,2}):(\d{1,2})(\.\d+)?$")

# ``HHMMSS`` with optional fraction: DICOM TM, and the tail of a DICOM DT.
_DICOM_TIME = re.compile(r"^(\d{2})(\d{2})(\d{2})(\.\d+)?$")


def parse_time_seconds(text: Optional[str]) -> Optional[float]:
    """Seconds since midnight, or ``None`` when the input says nothing.

    Accepts every form in the table above. Returns ``None`` rather than a
    guess for anything else, including a truncated ``HHMM``: a caller that
    orders acquisitions by this value is better off knowing the time is
    unavailable than being handed one that is two hours out.
    """
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    # ISO, with or without a date in front of it.
    m = _ISO_TIME.search(s)
    if m:
        return _seconds(m)

    # DICOM DT: strip the eight-digit date, leaving a TM.
    if len(s) >= 14 and s[:8].isdigit() and s[8:14].isdigit():
        s = s[8:]

    m = _DICOM_TIME.match(s)
    if m:
        return _seconds(m)

    return None


def _seconds(m: "re.Match[str]") -> Optional[float]:
    hours, minutes, seconds = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 61):
        return None
    frac = float(m.group(4)) if m.group(4) else 0.0
    return hours * 3600 + minutes * 60 + seconds + frac


__all__ = ["parse_time_seconds"]

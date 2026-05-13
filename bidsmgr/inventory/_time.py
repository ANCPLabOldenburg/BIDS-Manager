"""DICOM/BIDS time helpers shared across scan, fixups, and converter."""

from __future__ import annotations

from typing import Optional


def parse_dicom_time_seconds(text: Optional[str]) -> Optional[float]:
    """Convert DICOM TM (``HHMMSS.FFFFFF``) to seconds-since-midnight.

    Returns ``None`` for missing/empty/malformed input. Padding is applied to
    short strings (DICOM allows truncated forms like ``HHMM``); fractional
    seconds are preserved.
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        if "." in s:
            base, frac = s.split(".", 1)
        else:
            base, frac = s, "0"
        if len(base) < 6:
            base = base.ljust(6, "0")
        h = int(base[0:2])
        m = int(base[2:4])
        sec = int(base[4:6])
        return h * 3600 + m * 60 + sec + float("0." + frac)
    except (ValueError, IndexError):
        return None

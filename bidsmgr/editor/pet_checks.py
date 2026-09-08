"""PET consistency checks the BIDS schema cannot express.

A schema validator asks whether each field is present and well typed. It cannot
ask whether the fields agree with one another, because that is arithmetic and
domain knowledge rather than structure. These checks fill exactly that gap, and
they catch the errors that actually happen: a dose entered in the wrong unit, a
specific activity that does not follow from the dose and mass, frames that run
backwards, a reference time after the scan it is meant to precede.

Every finding is a warning, never an error. The BIDS spec permits all of these
files; what the checks say is "this looks like a data-entry mistake, please
look". Being wrong about that must cost the user a glance, not a blocked
conversion.

Qt-free. Consumed by the Editor's validation pane through the same adapter that
carries the TODO-placeholder findings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .types import Issue, Severity

# Conversion to megabecquerel, the unit BIDS PET datasets conventionally use.
_TO_MBQ: dict[str, float] = {
    "MBq": 1.0,
    "kBq": 1e-3,
    "Bq": 1e-6,
    "GBq": 1e3,
    "mCi": 37.0,
    "uCi": 0.037,
    "nCi": 3.7e-5,
    "Ci": 37000.0,
}

# A human injected dose outside this window is almost certainly a unit slip.
# Deliberately wide: it spans a low-dose amyloid scan through a high-dose
# whole-body FDG study, so only a genuine order-of-magnitude error trips it.
_PLAUSIBLE_MBQ = (0.1, 2000.0)

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _issue(message: str, field: str, *, rule: str) -> Issue:
    return Issue(
        severity=Severity.WARN,
        rule_id=f"bidsmgr.pet.{rule}",
        message=message,
        field=field,
        fix_label="Review",
        fix_action="set_field",
    )


def check_pet_sidecar(data: dict) -> list[Issue]:
    """Run every consistency check against one parsed PET sidecar."""
    out: list[Issue] = []
    out.extend(_check_dose_plausible(data))
    out.extend(_check_specific_radioactivity(data))
    out.extend(_check_frame_timing(data))
    out.extend(_check_time_zero(data))
    out.extend(_check_recon_parameters(data))
    return out


# ----------------------------------------------------------------------
# individual checks
# ----------------------------------------------------------------------


def _check_dose_plausible(data: dict) -> list[Issue]:
    """An injected dose far outside the human range means the units disagree."""
    dose = _as_float(data.get("InjectedRadioactivity"))
    units = str(data.get("InjectedRadioactivityUnits") or "").strip()
    if dose is None or dose <= 0 or not units:
        return []
    factor = _TO_MBQ.get(units)
    if factor is None:
        return []

    mbq = dose * factor
    low, high = _PLAUSIBLE_MBQ
    if low <= mbq <= high:
        return []
    return [_issue(
        f"InjectedRadioactivity is {dose} {units}, which is {mbq:.4g} MBq. "
        f"Typical human PET doses fall between {low} and {high:.0f} MBq, so "
        f"the value and its units may disagree.",
        "InjectedRadioactivity",
        rule="dose_implausible",
    )]


def _check_specific_radioactivity(data: dict) -> list[Issue]:
    """Specific activity should follow from dose over mass.

    Only checked when both sides are in units the check understands and the
    numbers are directly comparable, so a study using an exotic unit pair is
    left alone rather than warned about wrongly.
    """
    activity = _as_float(data.get("SpecificRadioactivity"))
    dose = _as_float(data.get("InjectedRadioactivity"))
    mass = _as_float(data.get("InjectedMass"))
    if activity is None or dose is None or mass is None or mass <= 0:
        return []
    if str(data.get("SpecificRadioactivityUnits") or "").strip() != "Bq/g":
        return []
    dose_units = str(data.get("InjectedRadioactivityUnits") or "").strip()
    if str(data.get("InjectedMassUnits") or "").strip() != "ug":
        return []
    factor = _TO_MBQ.get(dose_units)
    if factor is None:
        return []

    # dose (MBq -> Bq) over mass (ug -> g)
    expected = (dose * factor * 1e6) / (mass * 1e-6)
    if expected <= 0:
        return []
    ratio = activity / expected
    if 0.5 <= ratio <= 2.0:
        return []
    return [_issue(
        f"SpecificRadioactivity is {activity} Bq/g, but InjectedRadioactivity "
        f"over InjectedMass gives {expected:.4g} Bq/g. One of the three values "
        f"or their units is likely wrong.",
        "SpecificRadioactivity",
        rule="specific_activity_mismatch",
    )]


def _check_frame_timing(data: dict) -> list[Issue]:
    """Frame times must be monotonic, match in length, and start at zero.

    The last of those is the one worth flagging most often: BIDS defines
    FrameTimesStart relative to TimeZero, so a series that starts at a nonzero
    offset with no TimeZero recorded is ambiguous rather than wrong.
    """
    starts = data.get("FrameTimesStart")
    durations = data.get("FrameDuration")
    out: list[Issue] = []

    if isinstance(starts, list) and isinstance(durations, list):
        if len(starts) != len(durations):
            out.append(_issue(
                f"FrameTimesStart has {len(starts)} entries but FrameDuration "
                f"has {len(durations)}. There should be one of each per frame.",
                "FrameTimesStart",
                rule="frame_count_mismatch",
            ))

    if isinstance(starts, list) and len(starts) > 1:
        numeric = [_as_float(v) for v in starts]
        if all(v is not None for v in numeric):
            if any(b <= a for a, b in zip(numeric, numeric[1:])):
                out.append(_issue(
                    "FrameTimesStart is not strictly increasing, so the frames "
                    "are out of order or a value is mistyped.",
                    "FrameTimesStart",
                    rule="frame_times_not_monotonic",
                ))

    if isinstance(starts, list) and starts:
        first = _as_float(starts[0])
        if first is not None and first != 0 and "TimeZero" not in data:
            out.append(_issue(
                f"FrameTimesStart begins at {first} rather than 0, and no "
                f"TimeZero is recorded. BIDS defines frame times relative to "
                f"TimeZero, so the offset cannot be interpreted as it stands.",
                "FrameTimesStart",
                rule="frame_start_offset",
            ))

    return out


def _check_time_zero(data: dict) -> list[Issue]:
    """ScanStart and InjectionStart are offsets from TimeZero, so they need it."""
    out: list[Issue] = []
    time_zero = data.get("TimeZero")

    if time_zero is not None:
        text = str(time_zero).strip()
        if text and not _TIME_RE.match(text):
            out.append(_issue(
                f"TimeZero is {text!r}, which is not a hh:mm:ss time.",
                "TimeZero",
                rule="time_zero_format",
            ))

    for field in ("ScanStart", "InjectionStart"):
        value = _as_float(data.get(field))
        if value is not None and value != 0 and time_zero is None:
            out.append(_issue(
                f"{field} is {value}, an offset from TimeZero, but TimeZero is "
                f"not recorded.",
                field,
                rule="offset_without_time_zero",
            ))
    return out


def _check_recon_parameters(data: dict) -> list[Issue]:
    """The three reconstruction lists are positional and must align."""
    labels = data.get("ReconMethodParameterLabels")
    values = data.get("ReconMethodParameterValues")
    units = data.get("ReconMethodParameterUnits")
    lengths = {
        name: len(v)
        for name, v in (
            ("ReconMethodParameterLabels", labels),
            ("ReconMethodParameterValues", values),
            ("ReconMethodParameterUnits", units),
        )
        if isinstance(v, list)
    }
    if len(lengths) < 2 or len(set(lengths.values())) == 1:
        return []
    detail = ", ".join(f"{k} has {n}" for k, n in lengths.items())
    return [_issue(
        f"The reconstruction parameter lists have different lengths ({detail}). "
        f"They are positional, so entry N of each must describe the same "
        f"parameter.",
        "ReconMethodParameterValues",
        rule="recon_parameter_length_mismatch",
    )]


# ----------------------------------------------------------------------
# entry point used by the validator adapter
# ----------------------------------------------------------------------


def pet_issues_for(fp: Path) -> list[Issue]:
    """Consistency findings for a PET sidecar, or nothing for any other file."""
    name = fp.name.lower()
    if not name.endswith(".json") or "_pet" not in name:
        return []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    return check_pet_sidecar(data)


__all__ = ["check_pet_sidecar", "pet_issues_for"]

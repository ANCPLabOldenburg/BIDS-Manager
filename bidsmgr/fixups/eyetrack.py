"""The screen an eye tracker measured gaze on, in the run's events sidecar.

Gaze recorded "on screen" (an EyeLink's pixels) means nothing without the
screen: the standard puts it in the run's ``_events.json`` as
``StimulusPresentation`` (``ScreenResolution``, ``ScreenSize``,
``ScreenDistance``, ``ScreenOrigin``) and requires all four once the run has
events. So this runs AFTER the events are attached (``fixups/companion``),
and writes only beside an ``_events.tsv``: an events sidecar with no events
file is itself an error (``SIDECAR_WITHOUT_DATAFILE``).

The resolution is the file's; size, distance and origin are the user's
(``recording_meta.EyetrackSpec``). What is already in the sidecar stays.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _display(source: Path):
    """The screen resolution the tracker was set up with: the preamble's
    ``DISPLAY_COORDS``, else the recording's ``GAZE_COORDS`` (events only)."""
    from ..inventory.eyelink import read_preamble, read_recording

    display = read_preamble(source).get("display")
    if display:
        return display
    try:
        return read_recording(source, samples=False).info.get("display")
    except Exception:  # noqa: BLE001 - the resolution is then not stated
        return None


def place_stimulus_presentation(staging: Path, tasks) -> int:
    """Write ``StimulusPresentation`` for every eye-tracking task whose run
    has events; returns the sidecars written."""
    from ..converter.backends.eyelink import match_of, stimulus_presentation
    from ..util.paths import safe_path_component

    staging = Path(staging)
    written = 0
    for task in tasks:
        if task.modality != "eyetrack":
            continue
        folder = staging / (f"ses-{safe_path_component(task.session)}"
                            if task.session else "") / task.datatype
        events = folder / f"{match_of(task.basename)}_events.tsv"
        if not events.is_file():
            log.info("eye tracking: %s has no events file; the screen is recorded once "
                     "the run has one", task.basename)
            continue
        source = next((Path(f) for f in task.source_files if Path(f).exists()), None)
        display = _display(source) if source is not None else None
        screen = stimulus_presentation(display, dict(task.eyetrack or {}))
        sidecar = events.with_name(events.name[: -len(".tsv")] + ".json")
        data = {}
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        current = dict(data.get("StimulusPresentation") or {})
        merged = {**screen, **current}
        if merged == current:
            continue
        data["StimulusPresentation"] = merged
        sidecar.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        written += 1
    return written


__all__ = ["place_stimulus_presentation"]

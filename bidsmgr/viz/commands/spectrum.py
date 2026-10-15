"""Commands for spectra: processing and phasing."""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from . import command

if TYPE_CHECKING:  # pragma: no cover
    from ..store import SceneStore


def source(store: "SceneStore"):
    layer = store.scene.first("spectrum")
    if layer is None:
        return None
    return store.sources.get(layer.source)


@command("spectrum.set", "Spectrum processing", category="Spectrum", undoable=True)
def spectrum_set(
    store: "SceneStore",
    domain: Optional[Literal["spectrum", "fid"]] = None,
    part: Optional[Literal["magnitude", "real", "imaginary", "phase"]] = None,
    lb_hz: Optional[float] = None,
    phase0: Optional[float] = None,
    phase1_ms: Optional[float] = None,
    repeat: Optional[int] = None,
    edit: Optional[int] = None,
    metabolites: Optional[bool] = None,
    standard_window: Optional[bool] = None,
    reference: Optional[bool] = None,
    exclude_water: Optional[bool] = None,
    y_gain: Optional[float] = None,
) -> set[str]:
    """Patch how the spectrum is processed and shown."""
    sp = store.scene.spectrum
    src = source(store)
    values = dict(domain=domain, part=part, metabolites=metabolites,
                  standard_window=standard_window, reference=reference,
                  exclude_water=exclude_water)
    if y_gain is not None:
        values["y_gain"] = float(np.clip(y_gain, 1.0, 1000.0))
    if lb_hz is not None:
        values["lb_hz"] = float(np.clip(lb_hz, 0.0, 50.0))
    if phase0 is not None:
        # Wrapped into (-180, 180]: 190 degrees IS -170.
        values["phase0"] = float(((float(phase0) + 180.0) % 360.0) - 180.0) or 0.0
    if phase1_ms is not None:
        values["phase1_ms"] = float(np.clip(phase1_ms, -10.0, 10.0))
    if repeat is not None:
        n = src.n_dynamics if src is not None else 1
        values["repeat"] = -1 if int(repeat) < 0 else min(int(repeat), n - 1)
    if edit is not None:
        n = src.n_edits if src is not None else 1
        e = int(edit)
        values["edit"] = e if e in (-1, -2) else min(max(0, e), n - 1)
    changed = set()
    for key, value in values.items():
        if value is None:
            continue
        if getattr(sp, key) != value:
            setattr(sp, key, value)
            changed.add("spectrum")
    return changed


@command("spectrum.toggle", "Toggle a spectrum option", category="Spectrum")
def spectrum_toggle(store: "SceneStore",
                    field: Literal["fid", "metabolites", "standard_window", "reference",
                                   "exclude_water"]) -> set[str]:
    """Flip a switch: the FID page, the metabolite labels, the standard
    window, the reference overlay."""
    sp = store.scene.spectrum
    if field == "fid":
        return spectrum_set(store, domain="spectrum" if sp.domain == "fid" else "fid")
    return spectrum_set(store, **{field: not getattr(sp, field)})


@command("spectrum.phase_drag", "Phase the spectrum by dragging", category="Spectrum")
def spectrum_phase_drag(store: "SceneStore", d_phase0: float = 0.0,
                        d_phase1_ms: float = 0.0) -> set[str]:
    """Nudge the phases (a drag sends small steps)."""
    sp = store.scene.spectrum
    return spectrum_set(store, phase0=sp.phase0 + float(d_phase0),
                        phase1_ms=sp.phase1_ms + float(d_phase1_ms))


@command("spectrum.auto_phase", "Phase the spectrum automatically", category="Spectrum",
         undoable=True)
def spectrum_auto_phase(store: "SceneStore") -> set[str]:
    """The zero-order phase that makes NAA, creatine and choline upright and
    absorptive (``compute.mrs.auto_phase0``), on the selection shown."""
    from ..compute import mrs as M

    src = source(store)
    if src is None:
        return set()
    sp = store.scene.spectrum
    fid = src.select(dynamic=sp.repeat, edit=sp.edit)
    ppm, _hz, spec = M.spectrum(fid, src.dwell, src.spectrometer_mhz, src.nucleus,
                                phase1_ms=sp.phase1_ms)
    window = M.PHASE_WINDOW_1H if src.nucleus.upper() == "1H" else (float(ppm.min()),
                                                                     float(ppm.max()))
    return spectrum_set(store, phase0=M.auto_phase0(ppm, spec, window))


@command("spectrum.gain", "Scale the spectrum's height", category="Spectrum")
def spectrum_gain(store: "SceneStore", factor: float) -> set[str]:
    """Multiply the height gain (the wheel with Shift or Alt); 1 is the fit."""
    return spectrum_set(store, y_gain=store.scene.spectrum.y_gain * float(factor))


@command("spectrum.reset_processing", "Reset the processing", category="Spectrum",
         undoable=True)
def spectrum_reset(store: "SceneStore") -> set[str]:
    return spectrum_set(store, lb_hz=2.0, phase0=0.0, phase1_ms=0.0, repeat=-1, edit=-1)


__all__ = ["source"]

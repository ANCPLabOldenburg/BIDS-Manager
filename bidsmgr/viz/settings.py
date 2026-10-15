"""Every viewer preference, as one typed model.

One schema instead of scattered ``AppSettings`` keys and module globals: the
crosshair, the defaults a newly opened image gets, the 3-D quality, how traces
are drawn, the user's keyboard and mouse bindings, and the layouts used.
Stored as JSON under the ``viz/`` namespace (``gui/viz/settings_store.py``)
and broadcast on change, so an open viewer follows a preference at once.

Each field carries what the Settings page needs to draw it, as data: a
title, a help text, and in ``json_schema_extra`` the widget hints (``range``,
``step``, ``unit``, ``kind`` for a colour or a colour map, ``labels`` for a
choice). The Viewer page is GENERATED from this module, so a preference added
here appears in Settings without anyone building a control for it.

Pure data; validation clamps rather than rejects, so a hand-edited or older
settings file still loads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..qc.config import QcAnat, QcBold, QcConfig, QcDwi, QcMethods
from .render3d import QUALITY_DEFAULT, QUALITY_MAX, QUALITY_MIN


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


def _hint(**extra: Any) -> dict:
    return {"json_schema_extra": extra}


class CrosshairSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Crosshair")

    color: str = Field("#4FC3F7", title="Colour", **_hint(kind="colour"),
                       description="The crosshair on every slice and in the graph.")
    thickness: int = Field(1, title="Thickness", **_hint(range=(1, 5), unit="px"),
                           description="Line width of the crosshair.")
    gap: int = Field(0, title="Gap", **_hint(range=(0, 40), unit="px"),
                     description="Pixels left empty around the centre, so the "
                                 "voxel under the crosshair stays visible.")

    @field_validator("thickness")
    @classmethod
    def _thick(cls, v: int) -> int:
        return max(1, min(int(v), 5))

    @field_validator("gap")
    @classmethod
    def _gap(cls, v: int) -> int:
        return max(0, min(int(v), 40))


class VolumeSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Images")

    #: The layout a scan opens in: "" = the best this machine can show.
    mode: Literal["", "single", "multi", "3d", "combo", "hero", "mosaic"] = Field(
        "", title="Opens in",
        description="The layout a volume opens in. Automatic: the three "
                    "planes, with the 3-D view where there is a GPU. "
                    "Switching layout in the viewer changes this.",
        **_hint(labels={"": "Automatic", "single": "One plane",
                        "multi": "Multi-planar", "3d": "3-D",
                        "combo": "Planes + 3-D", "hero": "Large view",
                        "mosaic": "Mosaic"}))
    plane: Literal["sagittal", "coronal", "axial"] = Field(
        "axial", title="Plane",
        description="The plane of the one-plane layout, and the large view's.")
    gamma: float = Field(1.25, title="Gamma", **_hint(range=(0.1, 10.0), step=0.05),
                         description="Display gamma of a newly opened image: above 1 "
                                     "lifts the mid-tones.")
    interpolation: Literal["linear", "nearest"] = Field(
        "linear", title="Interpolation",
        description="Linear blends neighbouring voxels; nearest neighbour shows each "
                    "voxel as a square, as the data is.",
        **_hint(labels={"linear": "Linear (smooth)",
                        "nearest": "Nearest neighbour (voxels)"}))
    colormap: str = Field("gray", title="Colour map", **_hint(kind="colormap"),
                          description="The colour map a newly opened image gets.")
    labels: bool = Field(True, title="Orientation letters and cube")
    colorbar: bool = Field(False, title="Colour bar")
    ras: bool = Field(True, title="RAS orientation",
                      description="Off: each axis in the file's own storage order.")
    radiological: bool = Field(False, title="Radiological convention",
                               description="Mirror left and right: the patient's left "
                                           "on the image's right.")
    space: Literal["voxel", "world"] = Field(
        "voxel", title="Slices",
        description="Voxel: the file's own grid (exact, fast). World: the "
                    "scanner's axes, resampled, so an oblique scan is upright.",
        **_hint(labels={"voxel": "Voxel grid", "world": "World space"}))
    #: Frames kept in memory, MB. 0 = automatic (half the free memory).
    memory_mb: int = Field(0, title="Memory for 4-D series",
                           **_hint(range=(0, 262144), step=256, unit="MB", zero="Automatic"),
                           description="The most a 4-D series may hold in memory. "
                                       "Automatic is half of the free memory.")
    #: Playback speed, frames per second.
    fps: float = Field(8.0, title="Playback speed", **_hint(range=(0.5, 60.0), step=0.5,
                                                            unit="volumes/s"))
    inspector: bool = Field(True, title="Advanced controls open",
                            description="The column beside the images with the layers, "
                                        "their look, the view, the layout and the 3-D "
                                        "controls.")

    @field_validator("gamma")
    @classmethod
    def _gamma(cls, v: float) -> float:
        return max(0.1, min(float(v), 10.0))

    @field_validator("fps")
    @classmethod
    def _fps(cls, v: float) -> float:
        return max(0.5, min(float(v), 60.0))

    @field_validator("memory_mb")
    @classmethod
    def _mem(cls, v: int) -> int:
        return max(0, int(v))


class RenderSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, title="3-D")

    quality: int = Field(QUALITY_DEFAULT, title="Ray samples",
                         **_hint(range=(QUALITY_MIN, QUALITY_MAX), step=64, unit="samples"),
                         description="Samples along each ray of a newly opened rendering. "
                                     "More shows finer detail and renders slower.")

    @field_validator("quality")
    @classmethod
    def _q(cls, v: int) -> int:
        return max(QUALITY_MIN, min(int(v), QUALITY_MAX))


class TraceSettings(_Section):
    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Signals")

    #: 0 = automatic (thin for many traces, 2 px for one or two).
    line_width: int = Field(0, title="Line width", **_hint(range=(0, 8), unit="px",
                                                            zero="Automatic"),
                            description="Automatic: thin for many traces, 2 px for "
                                        "one or two.")
    line_color: str = Field("", title="Line colour", **_hint(kind="colour", empty="By type"),
                            description="One colour for every trace; empty colours "
                                        "each by its channel type.")
    type_colors: dict[str, str] = Field(default_factory=dict, title="Channel type colours")
    dark_plot: bool = Field(False, title="Dark plot in the light theme",
                            description="Draw signals on a dark canvas even when the "
                                        "app is light. In the dark theme the canvas is "
                                        "dark already.")
    event_width: int = Field(2, title="Event line width", **_hint(range=(1, 10), unit="px"))
    controls_open: bool = Field(False, title="Advanced controls open",
                                description="The column beside the traces with the channel, "
                                            "time, filter, event, QC and display controls.")
    event_color: str = Field("", title="Event colour",
                             **_hint(kind="colour", empty="By label"),
                             description="One colour for every event; empty colours "
                                         "each label differently.")

    @field_validator("line_width")
    @classmethod
    def _w(cls, v: int) -> int:
        v = int(v)
        return 0 if v <= 0 else min(v, 8)

    @field_validator("event_width")
    @classmethod
    def _ew(cls, v: int) -> int:
        return max(1, min(int(v), 10))


class QcSettings(_Section):
    """Quality control in every viewer."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="When QC runs")

    on_open: bool = Field(
        False, title="Run QC when a file opens",
        description="On: QC that was on stays on for the next file and is computed as "
                    "soon as it opens (the QC plots under a BOLD's time course, the MEG and "
                    "EEG check). Off: every file opens with QC off, and QC is computed "
                    "when you switch it on.")


class MeegQcSettings(_Section):
    """The quick MEG and EEG quality check (``compute.meeg_qc``). Every
    measure is taken within ONE channel type (magnetometers, gradiometers,
    EEG): types measure different things in different units, so a channel is
    only ever compared with channels of its own type."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="MEG and EEG QC")

    types: list[str] = Field(
        default_factory=list, title="Channel types checked",
        description="The channel types QC runs on (mag, grad, eeg, ...); empty: all of "
                    "them. Chosen per recording in the QC section.")
    segment_s: float = Field(
        2.0, title="Segment length", **_hint(range=(0.5, 30.0), step=0.5, unit="s"),
        description="The recording is checked in segments (epochs) of this length.")
    highpass_hz: float = Field(
        1.0, title="Ignore drifts below", **_hint(range=(0.0, 20.0), step=0.1, unit="Hz",
                                                 zero="Off"),
        description="A zero-phase high-pass filter applied before measuring: slow drifts "
                    "(the environment, sweat, electrode settling) differ from sensor to "
                    "sensor and would otherwise decide which channel looks noisy. 0 "
                    "measures the raw signal.")
    apply_proj: bool = Field(
        True, title="Apply the recording's SSP projectors",
        description="MEG recordings usually carry projectors that remove the room's "
                    "magnetic field; applied, a sensor is judged by what is left.")
    use_std: bool = Field(
        True, title="Use the standard deviation (STD)",
        description="Judge channels and segments by their standard deviation.")
    use_ptp: bool = Field(
        True, title="Use the peak-to-peak amplitude (PtP)",
        description="Judge channels and segments by their peak-to-peak range: catches "
                    "jumps and pops that barely move the standard deviation.")
    noisy_z: float = Field(
        3.0, title="Noisy channel above", **_hint(range=(1.0, 10.0), step=0.5,
                                                  unit="robust SD"),
        description="A channel is NOISY when its level (log STD or log PtP) is more than "
                    "this many robust standard deviations above the other channels of "
                    "its type.")
    flat_ratio: float = Field(
        0.01, title="Flat channel below", **_hint(range=(0.001, 0.5), step=0.001,
                                                 unit="of its type"),
        description="A channel is FLAT when its level is below this fraction of its "
                    "type's median (0.01: a hundred times quieter than the rest).")
    min_correlation: float = Field(
        0.6, title="Channels follow their type above", **_hint(range=(0.0, 0.99), step=0.05,
                                                              unit="correlation", zero="Off"),
        description="How well a channel correlates with the other channels of its type "
                    "(98th percentile, the PREP pipeline's measure). A loud channel that "
                    "follows them is picking up real fields and is not called noisy; one "
                    "below this, at a normal level, is called uncorrelated. 0 judges by "
                    "level alone.")
    line_db: float = Field(
        10.0, title="Line noise above", **_hint(range=(1.0, 40.0), step=1.0, unit="dB"),
        description="A channel has LINE NOISE when its power at the line frequency is "
                    "this much above its type's typical. Needs the recording's "
                    "PowerLineFrequency.")
    segment_factor: float = Field(
        4.0, title="Channel off in a segment beyond", **_hint(range=(1.5, 20.0), step=0.5,
                                                              unit="x its level"),
        description="In a segment, a channel is OFF when its STD or PtP there is more "
                    "than this many times its own usual level, or less than its inverse.")
    segment_share: float = Field(
        0.2, title="Segment flagged when off on", **_hint(range=(0.01, 1.0), step=0.01,
                                                         unit="of a type"),
        description="A segment is flagged when at least this share of the channels of "
                    "ONE type are off in it.")
    muscle: bool = Field(
        True, title="Look for muscle",
        description="The muscle band's share of the power, z-scored over time per "
                    "channel and averaged per type (after MNE's annotate_muscle_zscore). "
                    "Needs a sampling rate above twice the band's top.")
    muscle_low: float = Field(110.0, title="Muscle band from",
                              **_hint(range=(20.0, 500.0), step=5.0, unit="Hz"))
    muscle_high: float = Field(140.0, title="Muscle band to",
                               **_hint(range=(30.0, 1000.0), step=5.0, unit="Hz"))
    muscle_z: float = Field(
        4.0, title="Muscle above", **_hint(range=(1.0, 20.0), step=0.5, unit="z"),
        description="A segment is flagged for muscle when its type's mean z is above this.")


class VizSettings(_Section):
    crosshair: CrosshairSettings = Field(default_factory=CrosshairSettings)
    volume: VolumeSettings = Field(default_factory=VolumeSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    traces: TraceSettings = Field(default_factory=TraceSettings)
    qc: QcSettings = Field(default_factory=QcSettings)
    #: The MRI quality checks (``bidsmgr.qc.config``): methods and thresholds.
    qc_methods: QcMethods = Field(default_factory=QcMethods)
    qc_bold: QcBold = Field(default_factory=QcBold)
    qc_anat: QcAnat = Field(default_factory=QcAnat)
    qc_dwi: QcDwi = Field(default_factory=QcDwi)
    meeg_qc: MeegQcSettings = Field(default_factory=MeegQcSettings)
    #: action id -> key sequences, overriding the defaults (empty list = unbound).
    keymap: dict[str, list[str]] = Field(default_factory=dict)
    #: "<canvas>:<gesture>" -> tool id, overriding the default mouse map.
    mousemap: dict[str, str] = Field(default_factory=dict)
    #: "<layout>.<splitter>" -> splitter sizes, restored the next time.
    layout_sizes: dict[str, list[int]] = Field(default_factory=dict)
    #: Named views the user saved ("Save view as..."): JSON-able view state
    #: (layout, plane, display, graph, 3-D look), never a file or a cursor.
    view_presets: dict[str, dict] = Field(default_factory=dict)
    #: Per kind of file (a layout preset id such as ``mri.func``), the view
    #: it last had, so a BOLD run opens with its graph and a T1 without.
    layout_state: dict[str, dict] = Field(default_factory=dict)
    #: Which sections of the controls column are open (section key -> open).
    inspector_sections: dict[str, bool] = Field(default_factory=dict)
    #: The look every image opens with (display conventions, 3-D effect and
    #: parameters, clip planes), as last left: ``viz.memory``.
    volume_look: dict = Field(default_factory=dict)
    #: Per kind of signal (``meeg``, ``physio``), the trace options and
    #: filters as last left.
    traces_state: dict[str, dict] = Field(default_factory=dict)
    #: The spectrum's processing and display options as last left.
    spectrum_state: dict = Field(default_factory=dict)
    #: The spectroscopy viewer's controls column is open (it holds the
    #: processing, so it opens by default).
    spectrum_controls: bool = True


#: The sections the generated Viewer page shows, in order.
VIEWER_SECTIONS = ("crosshair", "volume", "render", "traces")
#: The sections the generated Quality control page shows, in order.
QC_SECTIONS = ("qc", "qc_methods", "qc_bold", "qc_anat", "qc_dwi", "meeg_qc")


def qc_config(settings: VizSettings) -> QcConfig:
    """The MRI quality checks' configuration, as the checks take it."""
    return QcConfig(methods=settings.qc_methods, bold=settings.qc_bold,
                    anat=settings.qc_anat, dwi=settings.qc_dwi)


__all__ = [
    "CrosshairSettings", "MeegQcSettings", "QC_SECTIONS", "QcSettings", "RenderSettings",
    "TraceSettings", "VIEWER_SECTIONS",
    "VizSettings", "VolumeSettings", "qc_config",
]

"""The scene: what is shown, and how, as plain data.

Every viewer in the library draws a :class:`Scene` and changes it only through
named commands (:mod:`bidsmgr.viz.commands`). That single path is what lets
the same state be saved, restored, synced between two viewers, scripted,
undone, and bound to any key, without any of those features knowing how a
canvas draws.

Pure data, as the architecture guards require: pydantic models with no I/O
and no reference to Qt. Mutable on purpose. A command changes a field in
place and reports which paths it touched, so a crosshair drag at 60 Hz costs
one attribute write instead of a rebuilt tree; immutable copies bought nothing
here but allocation.

Two vocabularies are worth fixing once, because the old viewer mixed them:

* a PLANE is which slice a 2-D view cuts (sagittal, coronal, axial);
* the WORLD is the scanner's coordinate system, in millimetres (RAS+: x grows
  toward the subject's right, y toward the front, z toward the top). The
  cursor lives in the world, never in one image's voxel grid, because the
  world is the only thing two images of the same head genuinely share.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

Vec3 = tuple[float, float, float]
Plane = Literal["sagittal", "coronal", "axial"]
PLANES: tuple[str, str, str] = ("sagittal", "coronal", "axial")

#: Which RAS axis a plane holds constant. Sagittal cuts at a fixed x.
PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}

#: For each plane, the RAS axes along the screen's horizontal and vertical.
#: Horizontal grows to the screen's right toward +axis (neurological view:
#: the subject's left on the image's left); vertical grows toward the TOP of
#: the screen. Sagittal shows the face on the right.
PLANE_HV = {"sagittal": (1, 2), "coronal": (0, 2), "axial": (0, 1)}

ViewMode = Literal["single", "multi", "3d", "combo", "hero", "mosaic"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class LabelTable(_Model):
    """Names and colours for an integer label image (an atlas)."""

    labels: dict[int, str] = Field(default_factory=dict)
    colors: dict[int, tuple[int, int, int]] = Field(default_factory=dict)


class VolumeDisplay(_Model):
    """How one volume layer turns numbers into colours on a 2-D slice.

    ``window`` is in the DATA's own units (Hounsfield, t-values, raw counts),
    never a 0-1 fraction: a window you cannot read off the image is a window
    you cannot reproduce. ``None`` means "not chosen yet", and the viewer
    fills it with a robust range when the first frame arrives.
    """

    colormap: str = "gray"
    #: Colour map for negative values ("" = negative values are not drawn
    #: separately). Two-tailed statistics use this.
    colormap_negative: str = ""
    window: Optional[tuple[float, float]] = None
    #: Window for the negative tail, as magnitudes ``(threshold, saturation)``.
    window_negative: Optional[tuple[float, float]] = None
    #: Display gamma: the normalised value v is drawn as ``v ** (1 / gamma)``,
    #: so gamma > 1 lifts the mid-tones. 1.25 reproduces the look the old
    #: viewer had through a hidden ``** 0.8``.
    gamma: float = 1.25
    invert: bool = False
    opacity: float = 1.0
    #: How values below the window's low end are drawn. ``range``: like any
    #: other value (the bottom colour), right for an anatomical background.
    #: ``hide_below``: transparent, right for a thresholded overlay.
    #: ``translucent_below``: faint, so a cluster that nearly reached the
    #: threshold still shows.
    threshold_mode: Literal["range", "hide_below", "translucent_below"] = "range"
    outline_px: float = 0.0
    interpolation: Literal["linear", "nearest"] = "linear"
    label_table: Optional[LabelTable] = None


class VolumeLayer(_Model):
    """A volume drawn in the scene: a source plus how to show it."""

    kind: Literal["volume"] = "volume"
    id: str
    source: str
    name: str = ""
    visible: bool = True
    #: Which frame of a 4-D (or 5-D, flattened) series is shown.
    frame: int = 0
    display: VolumeDisplay = Field(default_factory=VolumeDisplay)
    #: Whether the 3-D render includes this layer.
    in_3d: bool = True
    #: How a layer computed rather than read came about (``qc:tsnr``), so a
    #: saved scene can compute it again. Empty for a file.
    origin: str = ""


class SignalLayer(_Model):
    """A multichannel time series (MEG, EEG, physio) drawn as traces."""

    kind: Literal["signal"] = "signal"
    id: str
    source: str
    name: str = ""
    visible: bool = True


class SpectrumLayer(_Model):
    """A spectroscopy FID drawn as a spectrum."""

    kind: Literal["spectrum"] = "spectrum"
    id: str
    source: str
    name: str = ""
    visible: bool = True


Layer = Annotated[Union[VolumeLayer, SignalLayer, SpectrumLayer],
                  Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Views and display state
# ---------------------------------------------------------------------------


class Cursor(_Model):
    """Where the user is looking: a point in the world, a moment in time."""

    world: Optional[Vec3] = None
    time: Optional[float] = None


class Display(_Model):
    """Options every 2-D view of a volume shares."""

    #: True: each axis is drawn in RAS order whatever order the file stores
    #: it in. False: the file's own storage direction (a stored-reversed axis
    #: stays reversed), which is what the "RAS" toggle has always switched.
    ras: bool = True
    #: Mirror left and right (patient left on the image's right).
    radiological: bool = False
    #: ``voxel``: slices of the image's own grid, turned to the nearest
    #: anatomical axes (exact, fast). ``world``: slices along the scanner's
    #: axes, resampled, so an oblique or gantry-tilted image is upright.
    space: Literal["voxel", "world"] = "voxel"
    labels: bool = True
    colorbar: bool = False
    ruler: bool = False
    crosshair: bool = True


class SliceViewState(_Model):
    """Zoom and pan of one 2-D view. ``pan`` is in screen millimetres."""

    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)


class GraphState(_Model):
    """The 4-D time-course graph under the crosshair."""

    #: Neighbourhood: 1 = the voxel, 2 = 3x3, 3 = 5x5, 4 = 7x7.
    scope: int = 1
    dot: int = 8
    mark_neighbors: bool = True
    scaling: Literal["raw", "percent", "demean"] = "raw"
    #: ``auto`` picks seconds whenever the frames have a known time.
    x_axis: Literal["auto", "frames", "seconds"] = "auto"
    #: The run's ``_events.tsv`` drawn on the time axis.
    events: bool = True
    #: The run's physio recordings drawn under the graph, on the same axis.
    physio: bool = False
    #: Per-volume quality control under the graph (computed once per
    #: series, each row only when it is shown).
    qc: bool = False
    #: Which QC rows (ids of ``compute.qc.QC_ROWS``), IN THE ORDER the user
    #: put them: framewise displacement, translation, rotation, DVARS,
    #: outlier voxels, slice spikes, the global signal, the carpet.
    qc_rows: list[str] = Field(default_factory=lambda: ["fd", "translation", "rotation",
                                                        "dvars", "outliers"])
    #: The plots under the graph share its room (``fit``) or each has a
    #: readable height in a scrolling column (``scroll``).
    tracks_mode: Literal["fit", "scroll"] = "fit"

    @field_validator("qc_rows")
    @classmethod
    def _known_rows(cls, rows: list[str]) -> list[str]:
        """Only rows that exist, each once: a view saved by an earlier
        version names rows that no longer do (``motion``, which is now
        ``fd``, ``translation`` and ``rotation``), and one unknown id made
        every change to the rows fail."""
        from .compute.qc import QC_ROW_IDS

        return [r for r in dict.fromkeys(rows) if r in QC_ROW_IDS]
    #: The 4-D layer the graph and the volume controls follow ("" =
    #: automatic: the base image when it is a series, else the top-most
    #: 4-D overlay, so a BOLD drawn over a T1 has its time course).
    layer: str = ""


class LayoutState(_Model):
    """How the views of the multi-view layouts are arranged. The user's to
    change: which planes, in what order, in a row, a column or a grid, which
    view is the large one and how large, and where the graph sits."""

    #: ``auto``: whichever of row, column or grid makes the views largest
    #: for the window's shape (NiiVue's AUTO multiplanar layout).
    arrangement: Literal["auto", "row", "column", "grid"] = "auto"
    #: The planes of the three-plane layouts, in order (any of the three).
    planes: list[Plane] = Field(default_factory=lambda: ["sagittal", "coronal", "axial"])
    #: Hero layout: the large view (a plane or ``render``; "" = the plane
    #: the A / S / C keys chose).
    hero: str = ""
    #: Share of the window the large view takes.
    hero_fraction: float = 0.62
    hero_side: Literal["left", "top"] = "left"
    #: Where the time-course graph sits.
    graph: Literal["bottom", "right"] = "bottom"
    #: Where the quality panel sits: beside the views and the time course (a
    #: report is a tall list) or below them.
    quality: Literal["right", "bottom"] = "right"


class Camera(_Model):
    """The 3-D camera. Angles in radians, distance in box units."""

    az: float = -0.6
    el: float = 0.3
    dist: float = 1.9
    target: Vec3 = (0.0, 0.0, 0.0)


class ClipPlane(_Model):
    """One cutting plane of the 3-D render.

    ``pos`` and ``thick`` are fractions of the volume (0-1); azimuth and
    elevation are degrees. ``flip`` cuts from the other side.
    """

    active: bool = False
    az: float = 0.0
    el: float = 0.0
    pos: float = 0.5
    thick: float = 1.0
    flip: bool = False


class RenderState(_Model):
    """The 3-D render: effect, its parameters, camera.

    ``params`` holds, per effect, only the values the user changed away from
    that effect's baseline, in the slider units the presets use (the lighting
    material is the ``light`` parameter, so it too is remembered per effect).
    Switching effect therefore never loses a tweak, and a scene file stays
    small.
    """

    effect: str = "Standard"
    params: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: Changed values of the parameters every effect shares (whether the
    #: overlays are drawn, how much they show through): about the overlays,
    #: not the look, so switching effect must not change them.
    shared: dict[str, float] = Field(default_factory=dict)
    camera: Camera = Field(default_factory=Camera)
    #: How several clip planes combine. False: a point is cut when ANY plane
    #: cuts it (planes crop, six make a box). True: only when EVERY active
    #: plane cuts it, so the planes cut a wedge or a corner OUT.
    cut_away: bool = False
    #: Every active plane passes through the crosshair, keeping its angle:
    #: the cut faces are the slices the 2-D views show, and follow them.
    cut_at_cursor: bool = False
    #: The render coloured by the base image's 2-D colour map and window
    #: (the cut face then IS the slice). Off: the effect's own shading.
    use_colormap: bool = True


class Span(_Model):
    """A stretch of a recording marked bad: seconds of RUN time (as events
    are), and its label, ``BAD_`` and a reason (``BAD_muscle``), which is how
    mne-bids writes such a stretch into ``_events.tsv`` and reads it back as
    an annotation."""

    onset: float
    duration: float
    label: str = "BAD_"


class TraceFocus(_Model):
    """One channel over one stretch, outlined on the traces: where a click
    on a QC channel map took you. Seconds of RUN time, as a span's."""

    channel: str
    onset: float
    duration: float = 0.0


class TracesState(_Model):
    """How a signal is shown: which stretch, which channels, how scaled,
    how filtered. ``t0`` and ``width`` are seconds of RECORDING time."""

    t0: float = 0.0
    width: float = 10.0
    #: "all", "mag+grad", or a channel type.
    ch_type: str = "all"
    #: Channel names to show (None: every channel of ``ch_type``).
    picks: Optional[list[str]] = None
    #: Traces on screen at once, and the first of them.
    count: int = 20
    offset: int = 0
    scale: float = 1.0
    #: Each channel scaled to its own range (else one scale per type).
    normalize: bool = False
    hp: Optional[float] = None
    lp: Optional[float] = None
    notch: Optional[float] = None
    events: bool = False
    event_source: Literal["auto", "events.tsv", "stim", "annotations"] = "auto"
    #: Scale each page to itself (off: one amplitude per channel type for
    #: the whole recording, so pages compare).
    page_scale: bool = False
    #: Each channel's mean over the window taken out (MNE's remove_dc).
    remove_dc: bool = True
    #: A trace is clipped at 1.5 bands, so an artefact cannot paint over
    #: its neighbours (MNE's default clipping).
    clip: bool = True
    #: All channels of a type overlaid in one band.
    butterfly: bool = False
    #: The channels marked bad in this session (None: the recording's own,
    #: from its info and its _channels.tsv).
    bads: Optional[list[str]] = None
    #: The bad segments in this session (None: the recording's own, from its
    #: ``_events.tsv`` BAD rows, else its BAD annotations).
    bad_spans: Optional[list[Span]] = None
    #: Annotation mode: a drag marks a bad segment instead of scrolling.
    annotate: bool = False
    #: The label a new segment gets.
    annotate_label: str = "BAD_"
    #: The segment selected for editing (an index into the bad segments).
    selected_span: Optional[int] = None
    #: QC shown (computed when first switched on).
    quality: bool = False
    #: The QC plots under the traces: which measures (``meeg_qc.METRICS``),
    #: for which channel types (empty: every type checked), in the user's
    #: order (track ids), which the user hid, and whether they scroll at a
    #: readable height.
    qc_metrics: list[str] = Field(default_factory=lambda: ["off", "muscle"])
    qc_types: list[str] = Field(default_factory=list)
    qc_order: list[str] = Field(default_factory=list)
    qc_hidden: list[str] = Field(default_factory=list)
    qc_scroll: bool = False
    #: The QC plots to the right of the traces instead of under them.
    qc_beside: bool = False
    #: One channel over one stretch, outlined (``traces.go_to``); never
    #: remembered from one file to the next.
    focus: Optional[TraceFocus] = None


class SpectrumState(_Model):
    """How a spectrum is processed and shown."""

    domain: Literal["spectrum", "fid"] = "spectrum"
    #: The real part, phased, as every MRS package shows it. Magnitude needs
    #: no phase but its water tail is ~14x the real part's at 4.2 ppm.
    part: Literal["magnitude", "real", "imaginary", "phase"] = "real"
    lb_hz: float = 2.0
    phase0: float = 0.0
    #: First-order phase as the acquisition delay it undoes, in ms.
    phase1_ms: float = 0.0
    #: -1 = average of the repeats, else one repeat (0-based).
    repeat: int = -1
    #: -1 = average of the edit conditions, -2 = edit on minus off, else one.
    edit: int = -1
    metabolites: bool = True
    standard_window: bool = True
    reference: bool = False
    #: Fit the height leaving out the residual water band (1H): a water peak
    #: fifty times NAA would otherwise flatten every metabolite.
    exclude_water: bool = True
    #: Height gain over the fitted one: >1 looks under the tallest peak.
    y_gain: float = 1.0


class MosaicBuild(_Model):
    """A mosaic described by what a figure needs, not by its grammar line:
    one plane, a grid of slices across a range. The line
    (:attr:`Scene.mosaic`) is built from it; written by hand, the line is
    ``custom`` and the builder leaves it alone."""

    plane: Plane = "axial"
    rows: int = Field(3, ge=1, le=10)
    cols: int = Field(6, ge=1, le=16)
    #: Millimetres along the plane's axis; None: fitted to the head.
    start: Optional[float] = None
    end: Optional[float] = None
    labels: bool = True
    #: A perpendicular slice before the grid, showing where every tile cuts.
    reference: bool = False
    #: Neighbouring tiles overlap by this fraction.
    overlap: float = Field(0.0, ge=0.0, le=0.5)
    #: Tiles cropped to the head (one crop for all, so they stay alike).
    crop: bool = True
    #: The line was written by hand: the builder does not overwrite it.
    custom: bool = False


class Measurement(_Model):
    """A distance or angle drawn on a slice, kept in world millimetres."""

    kind: Literal["distance", "angle"]
    points: list[Vec3]
    plane: Plane


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------


class SourceRef(_Model):
    """Which file a source is. The path is spelled as given by the host."""

    id: str
    path: str
    kind: Literal["volume", "signal", "spectrum", "events"]


class Scene(_Model):
    """Everything about what one viewer shows."""

    #: Schema version of the serialised form.
    schema_version: int = 1
    sources: dict[str, SourceRef] = Field(default_factory=dict)
    layers: list[Layer] = Field(default_factory=list)
    cursor: Cursor = Field(default_factory=Cursor)

    # -- volumes -------------------------------------------------------
    mode: ViewMode = "multi"
    #: The plane of the single-view mode.
    plane: Plane = "axial"
    display: Display = Field(default_factory=Display)
    views: dict[str, SliceViewState] = Field(
        default_factory=lambda: {p: SliceViewState() for p in PLANES}
    )
    graph_visible: bool = False
    graph: GraphState = Field(default_factory=GraphState)
    layout: LayoutState = Field(default_factory=LayoutState)
    render: RenderState = Field(default_factory=RenderState)
    clips: list[ClipPlane] = Field(default_factory=lambda: [ClipPlane()])
    mosaic: str = "A -20 0 20 40 ; C 0 S X R 0"
    #: How the mosaic is built (plane, grid, range); the line follows.
    mosaic_build: MosaicBuild = Field(default_factory=MosaicBuild)
    measurements: list[Measurement] = Field(default_factory=list)

    # -- signals and spectra --------------------------------------------
    traces: TracesState = Field(default_factory=TracesState)
    spectrum: SpectrumState = Field(default_factory=SpectrumState)

    # ------------------------------------------------------------------
    def layer(self, layer_id: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def first(self, kind: str):
        """The first layer of ``kind`` ("signal", "spectrum", "volume")."""
        for layer in self.layers:
            if layer.kind == kind:
                return layer
        return None

    def base_layer(self) -> Optional[VolumeLayer]:
        """The bottom visible volume: the one whose grid the 2-D views cut."""
        for layer in self.layers:
            if layer.kind == "volume" and layer.visible:
                return layer
        for layer in self.layers:
            if layer.kind == "volume":
                return layer
        return None


__all__ = [
    "Camera", "ClipPlane", "Cursor", "Display", "GraphState", "LabelTable", "MosaicBuild",
    "Layer", "LayoutState", "Measurement", "PLANES", "PLANE_AXIS", "PLANE_HV", "Plane",
    "RenderState", "Scene", "SignalLayer", "SliceViewState", "SourceRef", "Span",
    "SpectrumLayer", "SpectrumState", "TracesState", "Vec3", "ViewMode",
    "VolumeDisplay", "VolumeLayer",
]

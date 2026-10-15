"""User-facing actions: what a button, a menu entry or a key does.

An action names a command and its parameters, a title, default keys, WHEN it
applies and, for a toggle, WHEN it is on. Toolbars, right-click menus, the
help overlay, the command palette and the keyboard are all built from this
one table, so a shortcut and the button it duplicates can never drift apart,
and the help can never list a key that does nothing (both happened with the
hand-written lists this replaces).

``when`` and ``checked`` are tiny expressions over a context dict the viewer
fills: atoms are flags (``gpu``) or ``key=value`` (``mode=multi``), joined by
``&&`` and ``||`` with ``!`` for negation. Kept deliberately small: anything
cleverer belongs in the context, where it can be tested.

Default keys are data. A user's bindings are stored as overrides
(``VizSettings.keymap``) and merged by :func:`effective_keymap`; conflicts are
reported per context by :func:`conflicts`, never silently resolved.

Pure data, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ActionDef:
    id: str
    title: str
    #: Command id to run, or ``gui:<name>`` for something only the GUI does
    #: (play, open the help, take a screenshot).
    command: str
    params: Mapping[str, Any] = field(default_factory=dict)
    category: str = "General"
    keys: tuple[str, ...] = ()
    when: str = ""
    checked: str = ""
    #: Whether its button appears at all (empty: always). ``when`` greys an
    #: action out where it does not apply NOW; ``shown`` hides one that can
    #: never apply to this kind of file (Fit all on a MEG recording).
    shown: str = ""
    help: str = ""
    #: Where it shows: "toolbar" actions get a button when a layout lists them.
    short: str = ""
    #: The icon its button and menu entry carry (a logical name of
    #: ``bidsmgr.gui.icons``), "" for none.
    icon: str = ""

    @property
    def checkable(self) -> bool:
        return bool(self.checked)

    @property
    def label(self) -> str:
        return self.short or self.title


def _a(id_, title, command, params=None, **kw) -> ActionDef:
    return ActionDef(id_, title, command, dict(params or {}), **kw)


#: Every action, in help / palette order.
ACTIONS: tuple[ActionDef, ...] = (
    # -- layout -------------------------------------------------------------
    _a("view.axial", "Axial view", "view.plane", {"plane": "axial", "single": True},
       category="Views", keys=("A",), when="volume", short="Axial",
       checked="mode=single && plane=axial", icon="layout_single"),
    _a("view.sagittal", "Sagittal view", "view.plane", {"plane": "sagittal", "single": True},
       category="Views", keys=("S",), when="volume", short="Sagittal",
       checked="mode=single && plane=sagittal", icon="layout_single"),
    _a("view.coronal", "Coronal view", "view.plane", {"plane": "coronal", "single": True},
       category="Views", keys=("C",), when="volume", short="Coronal",
       checked="mode=single && plane=coronal", icon="layout_single"),
    _a("view.multi", "Multi-planar view (three planes)", "view.toggle_mode",
       {"mode": "multi"}, category="Views", keys=("M",), when="volume",
       short="Multi-planar", checked="mode=multi",
       help="Sagittal, coronal and axial slices through the crosshair, side by side "
            "(multi-planar reconstruction, MPR)", icon="layout_multi"),
    _a("view.3d", "3-D volume rendering", "view.toggle_mode", {"mode": "3d"},
       category="Views", keys=("D",), when="volume.3d && gpu", short="3-D",
       checked="mode=3d",
       help="The whole volume ray-cast on the graphics card: surfaces, projections, "
            "clip planes", icon="layout_3d"),
    _a("view.combo", "Multi-planar view with 3-D rendering", "view.toggle_mode",
       {"mode": "combo"}, category="Views", keys=("P",), when="volume.3d && gpu",
       short="Planes + 3-D", checked="mode=combo",
       help="The three slices and the 3-D rendering together, one crosshair", icon="layout_combo"),
    _a("view.hero", "One large view, the others beside it", "view.toggle_mode",
       {"mode": "hero"}, category="Views", when="volume", short="Large view",
       checked="mode=hero",
       help="The chosen plane (or the 3-D) drawn large, the rest small beside it; drag "
            "the gap to resize", icon="layout_hero"),
    _a("view.mosaic", "Mosaic of many slices (lightbox)", "view.toggle_mode",
       {"mode": "mosaic"}, category="Views", when="volume", short="Mosaic",
       checked="mode=mosaic",
       help="Many slices of one plane in a grid, for a quick look through the volume "
            "or a figure", icon="layout_mosaic"),
    _a("view.graph", "Time-course graph", "view.graph", category="Views",
       keys=("G",), when="volume.4d", shown="volume.4d", short="Time course",
       checked="graph",
       help="The signal over time at the crosshair (4-D images), with the run's events, "
            "physiology and quality rows when asked", icon="timecourse"),
    _a("graph.beside", "Time course beside the views", "gui:graph_beside", category="Views",
       when="graph", shown="volume.4d", short="Beside", checked="graph.beside",
       help="Put the time-course panel to the right of the views instead of under them "
            "(a long run reads better wide, a tall one beside)", icon="dock_right"),
    _a("graph.maximize", "Maximise the time course", "gui:graph_maximize", category="Views",
       keys=("Ctrl+G",), when="graph && !graph.detached", shown="volume.4d",
       short="Maximise", checked="graph.maximized",
       help="Give the time-course panel the whole viewer (its QC plots and physiology "
            "too); again to bring the views back", icon="maximize"),
    _a("graph.detach", "Time course in its own window", "gui:graph_detach", category="Views",
       keys=("Ctrl+Shift+G",), when="graph", shown="volume.4d", short="Own window",
       checked="graph.detached",
       help="Move the time-course panel to a window of its own (a second screen, say); "
            "closing that window puts it back", icon="detach"),
    # -- display -----------------------------------------------------------
    _a("view.labels", "Orientation labels and cube", "view.flag", {"flag": "labels"},
       category="Display", keys=("O",), when="volume", short="Orientation labels",
       checked="display.labels",
       help="Letters at the edges of every slice (R/L, A/P, S/I) and the orientation "
            "cube in 3-D"),
    _a("view.ras", "RAS orientation", "view.flag", {"flag": "ras"},
       category="Display", when="volume", short="RAS", checked="display.ras",
       help="Draw every image Right, Anterior, Superior up the axes, whatever order its "
            "file stores the voxels in. Off: the file's own storage order"),
    _a("view.radiological", "Radiological convention", "view.flag",
       {"flag": "radiological"}, category="Display", keys=("L",), when="volume",
       short="Radiological", checked="display.radiological",
       help="Mirror left and right, as radiologists read images: the patient's left on "
            "the screen's right. Off: neurological convention"),
    _a("view.world", "World space", "view.space", category="Display", keys=("W",),
       when="volume", short="World space", checked="space=world",
       help="Resample slices onto the scanner's axes, so an oblique acquisition is "
            "drawn upright. Off: slices of the image's own voxel grid"),
    _a("view.colorbar", "Colour bar", "view.flag", {"flag": "colorbar"},
       category="Display", keys=("B",), when="volume", short="Colour bar",
       checked="display.colorbar",
       help="A colour bar under the image for every scalar layer, with its window, "
            "units and threshold"),
    _a("view.crosshair", "Crosshair", "view.flag", {"flag": "crosshair"},
       category="Display", keys=("X",), when="volume", short="Crosshair",
       checked="display.crosshair",
       help="The crosshair on every slice and in 3-D (its colour and width in Slice "
            "views)"),
    _a("view.reset", "Reset zoom and pan", "view.reset", category="Display",
       keys=("R",), when="volume", short="Reset zoom"),
    # -- navigation --------------------------------------------------------
    _a("slice.next", "Next slice", "gui:slice", {"n": 1}, category="Navigate",
       keys=("Up",), when="volume && slices"),
    _a("slice.prev", "Previous slice", "gui:slice", {"n": -1}, category="Navigate",
       keys=("Down",), when="volume && slices"),
    _a("slice.next10", "Ten slices forward", "gui:slice", {"n": 10},
       category="Navigate", keys=("PgUp",), when="volume && slices"),
    _a("slice.prev10", "Ten slices back", "gui:slice", {"n": -10},
       category="Navigate", keys=("PgDown",), when="volume && slices"),
    _a("frame.next", "Next volume", "frame.step", {"n": 1}, category="Navigate",
       keys=("Right",), when="volume.4d"),
    _a("frame.prev", "Previous volume", "frame.step", {"n": -1},
       category="Navigate", keys=("Left",), when="volume.4d"),
    _a("frame.first", "First volume", "frame.set", {"frame": 0},
       category="Navigate", keys=("Shift+Left",), when="volume.4d"),
    _a("frame.last", "Last volume", "frame.set", {"frame": 10 ** 9},
       category="Navigate", keys=("Shift+Right",), when="volume.4d"),
    _a("frame.same_shell", "Next volume of this shell", "frame.same_shell", {"n": 1},
       category="Diffusion", keys=("]",), when="volume.dwi",
       help="Step through every volume of the shell on screen: how a bad "
            "direction is found"),
    _a("frame.same_shell_prev", "Previous volume of this shell", "frame.same_shell",
       {"n": -1}, category="Diffusion", keys=("[",), when="volume.dwi"),
    _a("frame.shell", "Next shell", "frame.shell", {"n": 1}, category="Diffusion",
       when="volume.dwi", help="The first volume of the next b-value"),
    _a("frame.shell_prev", "Previous shell", "frame.shell", {"n": -1},
       category="Diffusion", when="volume.dwi"),
    # -- the dataset around the image ---------------------------------------
    _a("nav.run_next", "Next run", "gui:navigate", {"entity": "run", "step": 1},
       category="Dataset", keys=("Alt+Right",), when="nav.run.next",
       help="The same image of the next run, crosshair where it was"),
    _a("nav.run_prev", "Previous run", "gui:navigate", {"entity": "run", "step": -1},
       category="Dataset", keys=("Alt+Left",), when="nav.run.prev"),
    _a("nav.echo_next", "Next echo", "gui:navigate", {"entity": "echo", "step": 1},
       category="Dataset", when="nav.echo.next"),
    _a("nav.echo_prev", "Previous echo", "gui:navigate", {"entity": "echo", "step": -1},
       category="Dataset", when="nav.echo.prev"),
    _a("nav.ses_next", "Next session", "gui:navigate", {"entity": "ses", "step": 1},
       category="Dataset", keys=("Alt+PgDown",), when="nav.ses.next"),
    _a("nav.ses_prev", "Previous session", "gui:navigate", {"entity": "ses", "step": -1},
       category="Dataset", keys=("Alt+PgUp",), when="nav.ses.prev"),
    _a("nav.sub_next", "Next subject", "gui:navigate", {"entity": "sub", "step": 1},
       category="Dataset", keys=("Alt+Down",), when="nav.sub.next",
       help="The same image of the next subject"),
    _a("nav.sub_prev", "Previous subject", "gui:navigate", {"entity": "sub", "step": -1},
       category="Dataset", keys=("Alt+Up",), when="nav.sub.prev"),
    _a("deface.preview", "Preview defacing", "gui:deface_preview", category="Layers",
       when="volume.deface",
       help="What the defacing engine would blank, drawn in red over the image. "
            "Nothing is changed"),
    _a("frame.play", "Play the series", "gui:play", category="Navigate",
       keys=("Space",), when="volume.4d", shown="volume.4d", short="Play", checked="playing", icon="play"),
    _a("cursor.center", "Centre the crosshair", "cursor.center", category="Navigate",
       keys=("Home",), when="volume"),
    # -- window -------------------------------------------------------------
    _a("window.robust", "Window to the robust range", "window.robust",
       category="Contrast", keys=("F",), when="volume", short="Auto",
       help="The display range from the 1st to the 99th percentile of this volume"),
    _a("window.series", "Window for the whole series", "window.robust",
       {"series": True}, category="Contrast", keys=("Shift+F",), when="volume.4d",
       help="One display range for every volume of the series, so playing it does not "
            "flicker"),
    _a("window.full", "Window to the full range", "window.full",
       category="Contrast", when="volume", short="Full",
       help="The display range from the smallest to the largest value"),
    _a("window.invert", "Invert the colour map", "layer.toggle",
       {"field": "invert"}, category="Contrast", keys=("I",), when="volume",
       checked="layer.invert"),
    _a("view.nearest", "Nearest-neighbour interpolation", "layer.toggle",
       {"field": "nearest"}, category="Display", keys=("N",), when="volume",
       short="Nearest neighbour", checked="layer.nearest",
       help="Draw each voxel as a square, as the data is, instead of smoothing "
            "between voxels"),
    # -- 3-D ----------------------------------------------------------------
    _a("clip.toggle", "Clip plane on or off", "clip.toggle", category="3-D slicer",
       keys=("Shift+Z",), when="render", checked="clip.active"),
    _a("clip.axial", "Axial cut", "clip.axis", {"plane": "axial"},
       category="3-D slicer", keys=("Shift+A",), when="render"),
    _a("clip.sagittal", "Sagittal cut", "clip.axis", {"plane": "sagittal"},
       category="3-D slicer", keys=("Shift+S",), when="render"),
    _a("clip.coronal", "Coronal cut", "clip.axis", {"plane": "coronal"},
       category="3-D slicer", keys=("Shift+C",), when="render"),
    _a("clip.invert", "Cut from the other side", "clip.invert",
       category="3-D slicer", keys=("Shift+X",), when="render"),
    _a("clip.at_cursor", "Cut through the crosshair", "clip.toggle_at_cursor",
       category="3-D slicer", keys=("Shift+T",), when="render", checked="clip.at_cursor",
       help="Every clip plane passes through the crosshair: the cut faces are "
            "the slices the 2-D views show, and follow them"),
    _a("clip.corner", "Corner cut at the crosshair", "clip.preset", {"preset": "corner"},
       category="3-D slicer", keys=("Shift+O",), when="render",
       help="Cut out the corner facing you, at the crosshair: the three slices "
            "in 3-D"),
    _a("render.reset_view", "Reset the 3-D camera", "render.reset_view",
       category="3-D", keys=("Shift+R",), when="render"),
    _a("render.left", "Look from the left", "render.preset_view", {"side": "left"},
       category="3-D", when="render"),
    _a("render.right", "Look from the right", "render.preset_view", {"side": "right"},
       category="3-D", when="render"),
    _a("render.front", "Look from the front", "render.preset_view", {"side": "front"},
       category="3-D", when="render"),
    _a("render.top", "Look from the top", "render.preset_view", {"side": "top"},
       category="3-D", when="render"),
    _a("header.show", "Header and sidecar...", "gui:header", category="Layers",
       keys=("H",), when="volume", short="Header",
       help="The image's header beside its sidecar, every disagreement said: "
            "repetition time, slice timing, b-values, units, orientation"),
    _a("overlay.add", "Add an overlay...", "gui:add_overlay", category="Layers",
       when="volume", short="Add overlay",
       help="Draw another image over this one: a statistical map, an atlas, a "
            "mask, another contrast. Its look follows from what it holds."),
    _a("qc.check", "Check image quality", "gui:check_quality", category="QC",
       when="qc.checkable", shown="qc.checkable", short="Check quality", checked="qc.panel",
       help="The fast quality check of an anatomical or diffusion image: noise, "
            "contrast, artefacts, tissues, coverage, the gradient table, motion and "
            "slice dropout, each finding with its evidence over the image. The air is "
            "measured only in images that are not defaced."),
    _a("qc.below", "Quality panel below the views", "gui:quality_below", category="QC",
       when="qc.panel && !qc.detached", short="Below", checked="qc.below",
       help="Put the quality panel under the views and the time course instead of beside "
            "them", icon="dock_bottom"),
    _a("qc.maximize", "Maximise the quality panel", "gui:quality_maximize", category="QC",
       when="qc.panel && !qc.detached", short="Maximise", checked="qc.maximized",
       help="Give the quality panel the room of the views (below: of the views and the "
            "time course); again to bring them back", icon="maximize"),
    _a("qc.detach", "Quality panel in its own window", "gui:quality_detach", category="QC",
       when="qc.panel", short="Own window", checked="qc.detached",
       help="Move the quality panel to a window of its own; closing that window puts it "
            "back", icon="detach"),
    _a("qc.noise", "Show the noise", "gui:show_noise", category="QC",
       checked="qc.noise", when="volume", short="Show the noise",
       help="The image windowed to its air, in colour: ghosts, ringing, motion and "
            "wrap-around show as structure where there should be noise. Again to put "
            "the look back."),
    _a("qc.mean", "Mean image of the series", "gui:qc", {"map": "mean"},
       category="QC", when="volume.4d && volume.loaded", shown="volume.4d", short="Mean image",
       help="Computed when chosen: the mean over every volume, as an overlay"),
    _a("qc.sd", "Standard deviation map", "gui:qc", {"map": "sd"},
       category="QC", when="volume.4d && volume.loaded", shown="volume.4d",
       short="Standard deviation",
       help="Computed when chosen: where the signal moves over time (motion at the "
            "edges, vessels, ghosts), as an overlay"),
    _a("qc.tsnr", "Temporal SNR map", "gui:qc", {"map": "tsnr"},
       category="QC", when="volume.4d && volume.loaded", shown="volume.4d",
       short="Temporal SNR",
       help="Computed when chosen: each voxel's mean over its standard deviation after "
            "removing slow drifts, as an overlay with a colour bar and how to read it"),
    _a("view.inspector", "Advanced controls", "gui:inspector", category="Views",
       keys=("Ctrl+I",), when="volume || traces || spectrum", short="Advanced controls",
       checked="panel.inspector",
       help="Every control, grouped by purpose: for images the layers and their look, "
            "the view, the layout and the 3-D; for signals the channels, time, filters, "
            "events, QC and display; for spectra what is shown, the processing, the "
            "reference marks and QC", icon="controls"),
    # -- signals (MEG, EEG, physio) -------------------------------------------
    _a("time.next", "Next page of time", "time.page", {"n": 1}, category="Signals",
       keys=("Right",), when="traces"),
    _a("time.prev", "Previous page of time", "time.page", {"n": -1}, category="Signals",
       keys=("Left",), when="traces"),
    _a("time.start", "Start of the recording", "time.start", category="Signals",
       keys=("Shift+Left",), when="traces", short="|<"),
    _a("time.end", "End of the recording", "time.end", category="Signals",
       keys=("Shift+Right",), when="traces", short=">|"),
    _a("time.zoom_in", "Shorter time window", "time.zoom", {"factor": 0.5},
       category="Signals", keys=("=",), when="traces"),
    _a("time.zoom_out", "Longer time window", "time.zoom", {"factor": 2.0},
       category="Signals", keys=("-",), when="traces"),
    _a("time.fit", "Fit the whole recording", "time.fit", category="Signals",
       keys=("F",), when="traces && fit", shown="physio", short="Fit all",
       help="Put the whole recording in the window at once"),
    _a("channels.next", "Next channel", "traces.scroll", {"n": 1}, category="Signals",
       keys=("Down",), when="traces"),
    _a("channels.prev", "Previous channel", "traces.scroll", {"n": -1}, category="Signals",
       keys=("Up",), when="traces"),
    _a("channels.page_next", "Next page of channels", "traces.page_channels", {"n": 1},
       category="Signals", keys=("PgDown",), when="traces"),
    _a("channels.page_prev", "Previous page of channels", "traces.page_channels",
       {"n": -1}, category="Signals", keys=("PgUp",), when="traces"),
    _a("traces.bigger", "Larger amplitude", "traces.scale", {"factor": 1.25},
       category="Signals", keys=("]",), when="traces"),
    _a("traces.smaller", "Smaller amplitude", "traces.scale", {"factor": 0.8},
       category="Signals", keys=("[",), when="traces"),
    _a("traces.normalize", "Normalise each channel", "traces.normalize",
       category="Signals", keys=("N",), when="traces", short="Normalise",
       checked="normalize",
       help="Each channel scaled to its own range, so none overlaps; off, "
            "channels of one type share a scale"),
    _a("traces.butterfly", "Butterfly: every channel of a type in one band",
       "traces.option", {"field": "butterfly"}, category="Signals", keys=("B",),
       when="traces", short="Butterfly", checked="traces.butterfly",
       help="Overlay the channels of each type, to see what they share"),
    _a("traces.clip", "Clip traces to their band", "traces.option", {"field": "clip"},
       category="Signals", when="traces", short="Clip traces", checked="traces.clip",
       help="Cut a trace off at one and a half bands, so an artefact cannot "
            "paint over its neighbours"),
    _a("traces.dc", "Remove each channel's mean", "traces.option", {"field": "remove_dc"},
       category="Signals", keys=("D",), when="traces", short="Remove DC",
       checked="traces.remove_dc",
       help="Centre each trace in its band (the mean over the window taken out)"),
    _a("traces.page_scale", "Scale each page to itself", "traces.option",
       {"field": "page_scale"}, category="Signals", when="traces", short="Scale per page",
       checked="traces.page_scale",
       help="Off: one amplitude per channel type for the whole recording, so a "
            "quiet page and a noisy one compare. On: each page fills its bands."),
    _a("traces.quality", "Quality control (QC)", "traces.quality", category="Signals",
       keys=("Q",), when="traces && meeg", shown="meeg", short="QC",
       checked="quality",
       help="A quick check before a full pipeline, computed when switched on and type by "
            "type (magnetometers, gradiometers and EEG are never mixed): channels that are "
            "noisy, flat, uncorrelated or carry line noise, by STD and peak-to-peak; and "
            "the segments where a type's channels go off or muscle shows. Its parameters "
            "are in the advanced controls", icon="qc"),
    _a("annotate.toggle", "Annotation mode", "annotate.mode", category="Annotation",
       keys=("A",), when="traces && meeg", shown="meeg", short="Annotate",
       checked="annotate",
       help="Mark bad segments: drag across the traces to mark one, drag its edges "
            "to adjust it, right-click it to relabel or delete it. Channel names "
            "mark channels bad in any mode.", icon="annotate"),
    _a("annotate.delete", "Delete the selected bad segment", "annotate.remove",
       category="Annotation", keys=("Delete", "Backspace"),
       when="traces && annotate && span.selected", short="Delete segment",
       help="Click a segment to select it (it is drawn stronger), then Delete"),
    _a("review.save", "Save the bad channels and segments to the dataset", "gui:save_review",
       category="Annotation", when="traces && meeg && review.changed && writable",
       shown="meeg",
       short="Save to dataset",
       help="Bad channels into the status column of _channels.tsv, bad segments as "
            "BAD_ rows of the run's _events.tsv (how mne-bids reads them back as "
            "annotations); one step in the Editor's history, so it can be undone", icon="save"),
    _a("traces.reset", "Reset the view", "traces.reset", category="Signals",
       keys=("R",), when="traces", short="Reset view",
       help="Back to the opening window, scale, channels and no filter", icon="restore"),
    _a("traces.reset_filters", "Remove the filters", "traces.reset_filters",
       category="Signals", keys=("Shift+R",), when="traces && filtered",
       short="No filter", help="Back to the unfiltered signal"),
    _a("events.toggle", "Show events", "events.show", category="Signals",
       keys=("E",), when="traces && events", short="Events", checked="events.on",
       help="Event markers from the run's events.tsv, or the stim channel's "
            "triggers; turning them on jumps to the first when none is in view"),
    _a("traces.psd", "Power spectrum", "gui:psd", category="Signals", keys=("P",),
       when="traces", short="PSD",
       help="Power spectral density (Welch) of the channels shown, of the raw or the "
            "filtered signal", icon="psd"),
    _a("traces.line", "Line thickness and colours", "gui:line", category="Signals",
       when="traces || spectrum", short="Line style",
       help="Trace width, and one colour or a colour per channel type"),
    _a("traces.together", "Every physio file of this run", "gui:together",
       category="Signals", when="physio && relatives", shown="physio",
       short="All of this run",
       checked="together",
       help="The cardiac trace, the belt and the trigger of one run on one "
            "clock: resampled onto the fastest grid, each at its own start time"),
    _a("view.zen", "Zen mode: only the traces", "gui:zen", category="Signals",
       keys=("Z",), when="traces", short="Zen", checked="zen",
       help="Hide the toolbars and the overview and leave the traces the whole "
            "pane; Z again brings them back", icon="zen"),
    _a("signal.close", "Close the signal", "gui:close_signal", category="Signals",
       when="traces && meeg", shown="meeg", short="Close", icon="close",
       help="Drop the signal and go back to the metadata"),
    # -- spectra (MRS) ------------------------------------------------------
    _a("spectrum.fid", "Show the FID", "spectrum.toggle", {"field": "fid"},
       category="Spectrum", keys=("T",),
       when="spectrum", short="FID", checked="spectrum.fid",
       help="The time-domain signal the scanner measured, against seconds"),
    _a("spectrum.metabolites", "Metabolite positions", "spectrum.toggle",
       {"field": "metabolites"}, category="Spectrum", keys=("L",),
       when="spectrum && proton", short="Metabolites", checked="spectrum.metabolites",
       help="Labelled lines where the main metabolites resonate: NAA 2.01, creatine 3.03, "
            "choline 3.22, myo-inositol 3.56 ppm, and others"),
    _a("spectrum.window", "Standard window (0.2 to 4.2 ppm)", "spectrum.toggle",
       {"field": "standard_window"}, category="Spectrum", keys=("W",), when="spectrum && proton",
       short="0.2 to 4.2 ppm", checked="spectrum.window",
       help="Frame the band where the proton metabolites are read"),
    _a("spectrum.reference", "Overlay the water reference", "spectrum.toggle",
       {"field": "reference"}, category="Spectrum", keys=("O",), when="spectrum && reference",
       short="Water reference", checked="spectrum.reference",
       help="The unsuppressed water acquisition of the same voxel (_mrsref), over the "
            "spectrum: its line width and frequency are the reference for quality"),
    _a("spectrum.anatomy", "Show the voxel on the anatomy", "gui:anatomy",
       category="Spectrum", when="spectrum && voxel", short="On anatomy",
       help="Where the spectrum was measured: the voxel outlined on this "
            "subject's anatomical image"),
    _a("spectrum.fit", "Fit the whole band", "gui:spectrum_fit", category="Spectrum",
       keys=("F",), when="spectrum", short="Fit all",
       help="Show the whole frequency range at a height that fits it"),
    _a("spectrum.reset", "Reset the view", "gui:spectrum_reset", category="Spectrum",
       keys=("R",), when="spectrum", short="Reset view"),
    _a("spectrum.reset_processing", "Reset the processing", "spectrum.reset_processing",
       category="Spectrum", keys=("Shift+R",), when="spectrum"),
    _a("spectrum.auto_phase", "Phase automatically", "spectrum.auto_phase",
       category="Spectrum", keys=("P",), when="spectrum && !spectrum.fid", short="Auto phase",
       help="The zero-order phase that makes NAA, creatine and choline upright"),
    _a("spectrum.water", "Fit the height without the water", "spectrum.toggle",
       {"field": "exclude_water"}, category="Spectrum", when="spectrum && proton",
       short="Ignore water", checked="spectrum.exclude_water",
       help="Leave the residual water band (4.4 to 5.0 ppm) out of the height, so a "
            "water peak cannot flatten the metabolites; it then runs off the top"),
    _a("spectrum.gain_up", "Vertical zoom in (look under the peaks)", "spectrum.gain",
       {"factor": 1.25},
       category="Spectrum", keys=("]",), when="spectrum"),
    _a("spectrum.gain_down", "Vertical zoom out", "spectrum.gain", {"factor": 0.8},
       category="Spectrum", keys=("[",), when="spectrum"),
    # -- saved views ----------------------------------------------------------
    _a("scene.save", "Save a scene in this dataset...", "gui:save_scene",
       category="Views", when="volume && writable",
       help="THIS image with its overlays in their looks, the crosshair and the layout, "
            "saved in the dataset (.bidsmgr/viz/scenes) to open again or share with "
            "someone who has the dataset"),
    _a("view.mosaic_figure", "Save a mosaic figure...", "gui:mosaic_figure",
       category="Views", when="volume && volume.loaded", icon="mosaic_figure",
       help="The mosaic (built in Controls > Mosaic) as a PNG at screen, print or poster "
            "resolution"),
    _a("view.command_line", "Show the command line...", "gui:command_line",
       category="Views", when="volume && volume.loaded",
       help="The bidsmgr-view call (or Python) that reproduces this view: to open "
            "it again, or render it to a PNG with no window"),
    _a("views.save", "Save the look as a preset...", "gui:save_view", category="Views",
       when="volume", short="Save preset",
       help="How the viewer looks and is arranged (layout, display, time course, 3-D "
            "look) under a name, to apply to ANY image later from Save > Apply a "
            "preset. Without saving, the viewer already keeps the look you leave it in"),
    # -- general ------------------------------------------------------------
    _a("edit.undo", "Undo a display change", "gui:undo", category="General",
       keys=("Ctrl+Z",), when="undo"),
    _a("edit.redo", "Redo a display change", "gui:redo", category="General",
       keys=("Ctrl+Shift+Z",), when="redo"),
    _a("export.screenshot", "Save a screenshot...", "gui:screenshot",
       category="General", keys=("Ctrl+Shift+S",), when="volume || traces || spectrum",
       short="Screenshot", icon="camera"),
    _a("view.restore_defaults", "Restore every viewer default...", "gui:restore_defaults",
       category="General", when="volume || traces || spectrum", icon="restore",
       help="Every viewer back as installed: layouts, display and 3-D look, clip planes, "
            "trace and spectrum options, panel sizes. Shortcuts and saved presets are "
            "kept"),
    _a("help.shortcuts", "Keyboard and mouse shortcuts", "gui:help",
       category="General", keys=("F1",), short="Shortcuts", icon="shortcuts"),
    _a("help.palette", "Find an action", "gui:palette", category="General",
       keys=("Ctrl+Shift+P",)),
)

ACTION_BY_ID: dict[str, ActionDef] = {a.id: a for a in ACTIONS}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _atom(atom: str, ctx: Mapping[str, Any]) -> bool:
    atom = atom.strip()
    if not atom:
        return True
    negate = atom.startswith("!")
    if negate:
        atom = atom[1:].strip()
    if "=" in atom:
        key, value = (s.strip() for s in atom.split("=", 1))
        result = str(ctx.get(key, "")) == value
    else:
        result = bool(ctx.get(atom, False))
    return (not result) if negate else result


def evaluate(expr: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate ``a && b || !c``-style expressions (no parentheses)."""
    expr = (expr or "").strip()
    if not expr:
        return True
    return any(
        all(_atom(term, ctx) for term in clause.split("&&"))
        for clause in expr.split("||")
    )


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------


def normalise_key(key: str) -> str:
    """One spelling per key sequence (``ctrl+shift+z`` -> ``Ctrl+Shift+Z``)."""
    parts = [p.strip() for p in key.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return ""
    order = {"Ctrl": 0, "Meta": 1, "Alt": 2, "Shift": 3}
    mods = []
    base = parts[-1]
    for p in parts[:-1]:
        cap = p.capitalize()
        if cap == "Control":
            cap = "Ctrl"
        if cap == "Cmd" or cap == "Command":
            cap = "Ctrl"   # Qt maps Ctrl to Command on macOS
        mods.append(cap)
    mods = sorted(set(mods), key=lambda m: order.get(m, 9))
    base = base if len(base) > 1 else base.upper()
    if base.lower() in ("pgup", "pageup"):
        base = "PgUp"
    if base.lower() in ("pgdown", "pagedown", "pgdn"):
        base = "PgDown"
    if len(base) > 1 and base not in ("PgUp", "PgDown"):
        base = base[0].upper() + base[1:]
    return "+".join(mods + [base])


def effective_keymap(overrides: Optional[Mapping[str, list[str]]] = None) -> dict[str, list[str]]:
    """Every action's keys: the defaults, replaced where the user rebound."""
    out: dict[str, list[str]] = {}
    for action in ACTIONS:
        keys = list(action.keys)
        if overrides and action.id in overrides:
            keys = list(overrides[action.id])
        out[action.id] = [normalise_key(k) for k in keys if normalise_key(k)]
    return out


#: Condition words only one kind of viewer ever sets. An action whose
#: conditions use one of them belongs to that viewer; anything else with a
#: condition is a volume action; an action with none belongs to every viewer.
_KIND_ATOMS = {
    "signal": frozenset({"traces", "meeg", "physio", "fit", "filtered", "events",
                         "events.on", "relatives", "normalize", "together",
                         "traces.butterfly", "traces.clip", "traces.remove_dc",
                         "traces.page_scale", "review.changed", "zen", "annotate",
                         "span.selected", "quality"}),
    "spectrum": frozenset({"spectrum", "spectrum.fid", "spectrum.metabolites",
                           "spectrum.reference", "spectrum.window", "spectrum.exclude_water",
                           "proton", "reference", "voxel"}),
}
_EVERY_VIEWER = frozenset({"undo", "redo"})
VIEWER_KINDS = ("volume", "signal", "spectrum")


def kinds_of(action: ActionDef) -> frozenset[str]:
    """The viewers (``volume``, ``signal``, ``spectrum``) an action belongs to."""
    import re

    atoms: set[str] = set()
    for expr in (action.when, action.shown, action.checked):
        atoms |= set(re.findall(r"[A-Za-z_][A-Za-z_0-9.]*", expr or ""))
    atoms -= _EVERY_VIEWER
    if not atoms:
        return frozenset(VIEWER_KINDS)
    for kind, vocab in _KIND_ATOMS.items():
        if atoms & vocab:
            return frozenset({kind})
    return frozenset({"volume"})


def _contexts_overlap(a: ActionDef, b: ActionDef) -> bool:
    # Two actions whose WHEN can never both hold may share a key (Shift+A cuts
    # in 3-D only, A switches planes only where slices show). Anything else
    # sharing a key is a conflict the user must see.
    probe_sets = (
        {"traces": True, "fit": True, "filtered": True, "events": True, "physio": True,
         "relatives": True, "meeg": True, "undo": True, "redo": True},
        {"spectrum": True, "proton": True, "reference": True, "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": True, "render": False, "mode": "single", "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": False, "render": True, "mode": "3d", "undo": True, "redo": True},
        {"volume": True, "volume.4d": True, "volume.3d": True, "gpu": True,
         "slices": True, "render": True, "mode": "combo", "undo": True, "redo": True},
    )
    return any(evaluate(a.when, ctx) and evaluate(b.when, ctx) for ctx in probe_sets)


def conflicts(keymap: Mapping[str, list[str]]) -> list[tuple[str, str, str]]:
    """``(key, action, other action)`` for every key bound twice where both
    actions can apply at the same time."""
    owners: dict[str, list[str]] = {}
    for action_id, keys in keymap.items():
        for key in keys:
            owners.setdefault(key, []).append(action_id)
    out = []
    for key, ids in owners.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ACTION_BY_ID.get(ids[i]), ACTION_BY_ID.get(ids[j])
                if a is None or b is None or _contexts_overlap(a, b):
                    out.append((key, ids[i], ids[j]))
    return out


__all__ = [
    "ACTIONS", "ACTION_BY_ID", "ActionDef", "VIEWER_KINDS", "conflicts",
    "effective_keymap", "evaluate", "kinds_of", "normalise_key",
]
